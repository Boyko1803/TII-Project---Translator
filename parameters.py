import torch

sourceFileName = 'en_bg_data/train.bg'
targetFileName = 'en_bg_data/train.en'
sourceDevFileName = 'en_bg_data/dev.bg'
targetDevFileName = 'en_bg_data/dev.en'

corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel'

#device = torch.device("cuda:0")
device = torch.device("cpu")


wordEmbeddingSize = 256
encoderHiddenSize = 256
encoderLayers = 4
decoderHiddenSize = 2 * encoderHiddenSize
decoderLayers = encoderLayers
attentionHiddenSize = 256
dropout = 0.3
projectionTransformSize = 1024

learning_rate = 0.001
clip_grad = 5.0
learning_rate_decay = 0.5

batchSize = 32

maxEpochs = 12
log_every = 40
test_every = 2000

max_patience = 3
max_trials = 5

# parameters for beam search
use_beam_search = True
branching_factor = 4
