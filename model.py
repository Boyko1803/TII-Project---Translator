#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2022/2023
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import torch

class AdditiveAttentionModule(torch.nn.Module):
    def __init__(self, encoderOutputSize, decoderOutputSize, weightsSize):
        super(AdditiveAttentionModule, self).__init__()
        
        self.encoderOutputSize = encoderOutputSize
        self.decoderOutputSize = decoderOutputSize
        self.weightsSize = weightsSize
        
        self.We = torch.nn.Linear(encoderOutputSize, weightsSize)
        self.Wd = torch.nn.Linear(decoderOutputSize, weightsSize, bias = False)
        self.v = torch.nn.Linear(weightsSize, 1, bias = False)
    
    def forward(self, encoderContexts, decoderHiddenVector, paddingMask = None):
        # encoderContexts - (inputSentLen, batchSize, encoderOutputSize)
        # decoderHiddenVector - (outputSentLen, batchSize, decoderHiddenSize)
        # paddingMask - (batchSize, inputSentLen)
        
        weightE = self.We(encoderContexts.unsqueeze(0))
        weightD = self.Wd(decoderHiddenVector.unsqueeze(1))
        sumWeights = torch.tanh(weightE + weightD)
        # sumWeights - (outputSentLen, inputSentLen, batchSize, weightsSize)
        
        vectorWeights = self.v(sumWeights).squeeze(3).transpose(0,2).transpose(1,2)
        if paddingMask is not None:
            vectorWeights = vectorWeights.masked_fill(paddingMask.unsqueeze(1), -float('inf'))
        vectorWeights = torch.nn.functional.softmax(vectorWeights, dim = 2)
        vectors = encoderContexts.transpose(0,1)
        # vectorWeights - (batchSize, outputSentLen, inputSentLen)
        # vectors - (batchSize, inputSentLen, encoderOutputSize)
        
        outputVector = torch.matmul(vectorWeights, vectors).transpose(0,1)
        # vectorWeights - (outputSentLen, batchSize, encoderOutputSize)
        
        return outputVector


class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind, unkTokenIdx, padTokenIdx):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w,unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
        # sentsTensor - (sentLen, batchSize)
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName):
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(fileName, map_location = device))
    
    def __init__(self, wordEmbeddingSize, encoderHiddenSize, encoderLayers, decoderHiddenSize, decoderLayers, attentionHiddenSize, dropout, projectionTransformSize, sourceWord2ind, targetWord2ind, startToken, endToken, unkToken, padToken):
        super(NMTmodel, self).__init__()
        
        # word2ind
        self.sourceWord2ind = sourceWord2ind
        self.targetWord2ind = targetWord2ind
        
        # ind2word
        self.targetInd2word = {targetWord2ind[w] : w for w in targetWord2ind.keys()}
        
        # Special tokens
        self.sourceStartTokenIdx = sourceWord2ind[startToken]
        self.sourceEndTokenIdx = sourceWord2ind[endToken]
        self.sourceUnkTokenIdx = sourceWord2ind[unkToken]
        self.sourcePadTokenIdx = sourceWord2ind[padToken]
        
        self.targetStartTokenIdx = targetWord2ind[startToken]
        self.targetEndTokenIdx = targetWord2ind[endToken]
        self.targetUnkTokenIdx = targetWord2ind[unkToken]
        self.targetPadTokenIdx = targetWord2ind[padToken]
        
        # Parameters
        self.wordEmbeddingSize = wordEmbeddingSize
        self.encoderHiddenSize = encoderHiddenSize
        self.encoderLayers = encoderLayers
        self.decoderHiddenSize = decoderHiddenSize
        self.decoderLayers = decoderLayers
        self.attentionHiddenSize = attentionHiddenSize
        self.dropout = dropout
        self.projectionTransformSize = projectionTransformSize
        
        # Encoder components
        self.sourceWordEmbed = torch.nn.Embedding(len(sourceWord2ind), wordEmbeddingSize)
        self.encoderLSTM = torch.nn.LSTM(wordEmbeddingSize, encoderHiddenSize, num_layers = encoderLayers, bidirectional = True)
        
        # Decoder components
        self.targetWordEmbed = torch.nn.Embedding(len(targetWord2ind), wordEmbeddingSize)
        self.decoderLSTM = torch.nn.LSTM(wordEmbeddingSize, decoderHiddenSize, num_layers = decoderLayers)
        self.lateAttention = AdditiveAttentionModule(2 * encoderHiddenSize, decoderHiddenSize, attentionHiddenSize)
        self.dropout = torch.nn.Dropout(dropout)
        self.transform = torch.nn.Linear(2 * encoderHiddenSize + decoderHiddenSize, projectionTransformSize)
        self.projection = torch.nn.Linear(projectionTransformSize, len(targetWord2ind))
        
    
    def encodeSource(self, source):
        sourcePadded = self.preparePaddedBatch(source, self.sourceWord2ind, self.sourceUnkTokenIdx, self.sourcePadTokenIdx)
        # sourcePadded - (inputSentLen, batchSize)
        
        sourceEmbedding = self.sourceWordEmbed(sourcePadded)
        # sourceEmbedding - (inputSentLen, batchSize, wordEmbeddingSize)
        
        sourceLengths = [len(s) for s in source]
        encoderOutputPacked, (encoderH, encoderC) = self.encoderLSTM(torch.nn.utils.rnn.pack_padded_sequence(sourceEmbedding, sourceLengths, enforce_sorted = False))
        encoderOutput,_ = torch.nn.utils.rnn.pad_packed_sequence(encoderOutputPacked)
        hiddenVectors = encoderH.view(2, self.encoderLayers, -1, self.encoderHiddenSize).transpose(0,1).transpose(1,2).flatten(2,3)
        hiddenStates = encoderC.view(2, self.encoderLayers, -1, self.encoderHiddenSize).transpose(0,1).transpose(1,2).flatten(2,3)
        # encoderOutput - (inputSentLen, batchSize, 2 * encoderHiddenSize)
        # hiddenVectors, hiddenStates - (encoderLayers, batchSize, 2 * encoderHiddenSize)
        
        return encoderOutput, hiddenVectors, hiddenStates, sourceLengths
    
    def forward(self, source, target):
        device = next(self.parameters()).device
        
        encoderContexts, encoderHiddenVectors, encoderHiddenStates, sourceLengths = self.encodeSource(source)
        # encoderContexts - (inputSentLen, batchSize, 2 * encoderHiddenSize)
        # encoderHiddenVectors, encoderHiddenStates - (encoderLayers, batchSize, 2 * encoderHiddenSize)
        
        decoderInitialH = encoderHiddenVectors
        decoderInitialC = encoderHiddenStates
        # decoderInitialH, decoderInitialC - (decoderLayers, batchSize, decoderHiddenSize) // В архтектурата има изискване decoderLayers = encoderLayers и decoderHiddenSize = 2 * encoderHiddenSize
        
        targetPadded = self.preparePaddedBatch(target, self.targetWord2ind, self.targetUnkTokenIdx, self.targetPadTokenIdx)
        # targetPadded - (inputSentLen, batchSize)
        
        targetEmbedding = self.targetWordEmbed(targetPadded[:-1])
        # targetEmbedding - (outputSentLen - 1, batchSize, wordEmbeddingSize)
        
        targetLengths = [len(t)-1 for t in target]
        decoderOutputPacked,_ = self.decoderLSTM(torch.nn.utils.rnn.pack_padded_sequence(targetEmbedding, targetLengths, enforce_sorted = False), (decoderInitialH, decoderInitialC))
        decoderOutput,_ = torch.nn.utils.rnn.pad_packed_sequence(decoderOutputPacked)
        # decoderOutput = (outputSentLen - 1, batchSize, decoderHiddenSize)
        
        paddingMask = torch.zeros(size=(encoderContexts.shape[1], encoderContexts.shape[0]), dtype = bool, device = device)
        for i in range(len(sourceLengths)):
            paddingMask[i, sourceLengths[i]:] = 1
        # paddingMask - (batchSize, inputSentLen)
        
        lateAttentionVector = self.lateAttention(encoderContexts, decoderOutput, paddingMask)
        # lateAttentionVector - (outputSentLen - 1, batchSize, 2 * encoderHiddenSize)
        
        outputVector = torch.cat([lateAttentionVector, decoderOutput], dim = 2)
        # outputVector - (outputSentLen - 1, batchSize, 2 * encoderHiddenSize + decoderHiddenSize)
        
        drop = self.dropout(outputVector)
        trans = torch.nn.functional.relu(self.transform(drop.flatten(0,1)))
        proj = self.projection(trans)
        real = targetPadded[1:].flatten(0,1)
        # proj - ((outputSentLen - 1) * batchSize, len(targetWord2ind))
        # real - ((outputSentLen - 1) * batchSize)
        
        H = torch.nn.functional.cross_entropy(proj,real,ignore_index=self.targetPadTokenIdx)
        return H

    def translateSentenceBeamSearch(self, sentence, limit=1000, branching=4):
        device = next(self.parameters()).device
        
        possibleTranslations = []
        finishedTranslations = []
        currentLength = 1
        currentFinished = 0
        
        encoderContexts, encoderHiddenVectors, encoderHiddenStates, _ = self.encodeSource([sentence])
        # encoderContexts - (1, 1, 2 * encoderHiddenSize)
        # encoderHiddenVectors, encoderHiddenStates - (encoderLayers, 1, 2 * encoderHiddenSize)
        
        decoderInitialH = encoderHiddenVectors.squeeze(1)
        decoderInitialC = encoderHiddenStates.squeeze(1)
        # decoderInitialH, decoderInitialC - (decoderLayers, decoderHiddenSize)
        
        initialTranslation = [self.targetStartTokenIdx]
        tokenToAdd = self.targetStartTokenIdx
        token = torch.tensor(tokenToAdd, dtype=torch.long, device=device)
        
        h = decoderInitialH
        c = decoderInitialC
        
        tokenEmbedding = self.targetWordEmbed(token).unsqueeze(0)
        # tokenEmbedding - (1, decoderHiddenSize)
        
        decoderOutput, (hNew, cNew) = self.decoderLSTM(tokenEmbedding, (h, c))
        # decoderOutput - (1, decoderHiddenSize)
        
        lateAttentionVector = self.lateAttention(encoderContexts, decoderOutput.unsqueeze(0))
        # lateAttentionVector - (1, 1, 2 * encoderHiddenSize)
        
        outputVector = torch.cat([lateAttentionVector.squeeze(0).squeeze(0), decoderOutput.squeeze(0)], dim = 0)
        # outputVector - (2 * encoderHiddenSize + decoderHiddenSize)
        
        drop = self.dropout(outputVector)
        trans = torch.nn.functional.relu(self.transform(drop))
        proj = self.projection(trans)
        # proj - (len(targetWord2ind))
        
        distribution = torch.nn.functional.softmax(proj, dim = 0)
        distribution[self.targetStartTokenIdx] = 0
        distribution[self.targetUnkTokenIdx] = 0
        distribution[self.targetPadTokenIdx] = 0
        
        values, indexes = torch.topk(distribution, branching, largest=True, dim=0)
        values = -torch.log(values)
        for i in range(len(indexes)):
            token = indexes[i]
            tokenToAdd = token.item()
            newTranslation = initialTranslation.copy()
            newTranslation.append(tokenToAdd)
            (hSen, cSen) = (hNew, cNew)
            crossEntropySpeed = values[i]
            possibleTranslations.append((newTranslation, crossEntropySpeed, token, tokenToAdd, hSen, cSen))
        
        while currentLength <= limit and currentFinished < branching:
            newPossibleTranslations = []
            
            for tr in possibleTranslations:
                oldTranslation, oldCrossEntropySpeed, token, tokenToAdd, h, c = tr
                
                if tokenToAdd == self.targetEndTokenIdx:
                    currentFinished += 1
                    finishedTranslations.append(tr)
                    continue
                
                tokenEmbedding = self.targetWordEmbed(token).unsqueeze(0)
                # tokenEmbedding - (1, decoderHiddenSize)
                
                decoderOutput, (hNew, cNew) = self.decoderLSTM(tokenEmbedding, (h, c))
                # decoderOutput - (1, decoderHiddenSize)
                
                lateAttentionVector = self.lateAttention(encoderContexts, decoderOutput.unsqueeze(0))
                # lateAttentionVector - (1, 1, 2 * encoderHiddenSize)
                
                outputVector = torch.cat([lateAttentionVector.squeeze(0).squeeze(0), decoderOutput.squeeze(0)], dim = 0)
                # outputVector - (2 * encoderHiddenSize + decoderHiddenSize)
                
                drop = self.dropout(outputVector)
                trans = torch.nn.functional.relu(self.transform(drop))
                proj = self.projection(trans)
                # proj - (len(targetWord2ind))
                
                distribution = torch.nn.functional.softmax(proj, dim = 0)
                distribution[self.targetStartTokenIdx] = 0
                distribution[self.targetUnkTokenIdx] = 0
                distribution[self.targetPadTokenIdx] = 0
                
                values, indexes = torch.topk(distribution, branching, largest=True, dim=0)
                values = -torch.log(values)
                for i in range(len(indexes)):
                    token = indexes[i]
                    tokenToAdd = token.item()
                    newTranslation = oldTranslation.copy()
                    newTranslation.append(tokenToAdd)
                    (hSen, cSen) = (hNew, cNew)
                    newCrossEntropySpeed = (oldCrossEntropySpeed * currentLength + values[i]) / (currentLength + 1)
                    newPossibleTranslations.append((newTranslation, newCrossEntropySpeed, token, tokenToAdd, hSen, cSen))
            
            newPossibleTranslations.sort(reverse = False, key = (lambda x: x[1]))
            remaining = branching - currentFinished
            possibleTranslations = newPossibleTranslations[0:remaining]
            currentLength += 1
        
        if currentFinished < branching:
            finishedTranslations.extend(possibleTranslations)
        
        finishedTranslations.sort(reverse = False, key = (lambda x: x[1]))
        bestTranslation = finishedTranslations[0][0]
        
        if bestTranslation[-1] == self.targetEndTokenIdx:
            bestTranslation = bestTranslation[:-1]
        bestTranslation = bestTranslation[1:]
        
        result = [self.targetInd2word[ind] for ind in bestTranslation]
        
        return result
    
    def translateSentenceGreedy(self, sentence, limit=1000):
        device = next(self.parameters()).device
        
        encoderContexts, encoderHiddenVectors, encoderHiddenStates, _ = self.encodeSource([sentence])
        # encoderContexts - (1, 1, 2 * encoderHiddenSize)
        # encoderHiddenVectors, encoderHiddenStates - (encoderLayers, 1, 2 * encoderHiddenSize)
        
        decoderInitialH = encoderHiddenVectors.squeeze(1)
        decoderInitialC = encoderHiddenStates.squeeze(1)
        # decoderInitialH, decoderInitialC - (decoderLayers, decoderHiddenSize)
        
        translation = [self.targetStartTokenIdx]
        tokenToAdd = self.targetStartTokenIdx
        token = torch.tensor(tokenToAdd, dtype=torch.long, device=device)
        
        h = decoderInitialH
        c = decoderInitialC
        while len(translation) <= limit and tokenToAdd != self.targetEndTokenIdx:
            tokenEmbedding = self.targetWordEmbed(token).unsqueeze(0)
            # tokenEmbedding - (1, decoderHiddenSize)
            
            decoderOutput, (hNew, cNew) = self.decoderLSTM(tokenEmbedding, (h, c))
            # decoderOutput - (1, decoderHiddenSize)
            
            lateAttentionVector = self.lateAttention(encoderContexts, decoderOutput.unsqueeze(0))
            # lateAttentionVector - (1, 1, 2 * encoderHiddenSize)
            
            outputVector = torch.cat([lateAttentionVector.squeeze(0).squeeze(0), decoderOutput.squeeze(0)], dim = 0)
            # outputVector - (2 * encoderHiddenSize + decoderHiddenSize)
            
            drop = self.dropout(outputVector)
            trans = torch.nn.functional.relu(self.transform(drop))
            proj = self.projection(trans)
            # proj - (len(targetWord2ind))
            
            distribution = torch.nn.functional.softmax(proj, dim = 0)
            distribution[self.targetStartTokenIdx] = 0
            distribution[self.targetUnkTokenIdx] = 0
            distribution[self.targetPadTokenIdx] = 0
            
            token = torch.argmax(distribution)
            
            tokenToAdd = token.item()
            translation.append(tokenToAdd)
            (h, c) = (hNew, cNew)
        
        if translation[-1] == self.targetEndTokenIdx:
            translation = translation[:-1]
        translation = translation[1:]
        
        result = [self.targetInd2word[ind] for ind in translation]
        
        return result
    
    def translateSentence(self, sentence, limit=1000, beam_search = True, branching = 4):
        if beam_search:
            return self.translateSentenceBeamSearch(sentence, limit=limit, branching=branching)
        else:
            return self.translateSentenceGreedy(sentence, limit=limit)