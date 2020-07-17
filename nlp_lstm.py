
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.utils import download
import math
import gluonnlp as nlp


context = [mx.cpu()]


batch_size = 20 * len(context)
lr = 20
epochs = 3
bptt = 35
grad_clip = 0.25



# Specify the loss function, in this case, cross-entropy with softmax.
loss = gluon.loss.SoftmaxCrossEntropyLoss()


def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(i) for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def evaluate(model, data_source, batch_size, ctx):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(
        batch_size=batch_size, func=mx.nd.zeros, ctx=ctx)
    for i, (data, target) in enumerate(data_source):
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        output, hidden = model(data, hidden)
        hidden = detach(hidden)
        L = loss(output.reshape(-3, -1), target.reshape(-1))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal


dataset_name = 'wikitext-2'

# Load the dataset
train_dataset, val_dataset, test_dataset = [
    nlp.data.WikiText2(
        segment=segment, bos=None, eos='<eos>', skip_empty=False)
    for segment in ['train', 'val', 'test']
]

# Extract the vocabulary and numericalize with "Counter"
vocab = nlp.Vocab(
    nlp.data.Counter(train_dataset), padding_token=None, bos_token=None)

# Batchify for BPTT
bptt_batchify = nlp.data.batchify.CorpusBPTTBatchify(
    vocab, bptt, batch_size, last_batch='discard')
train_data, val_data, test_data = [
    bptt_batchify(x) for x in [train_dataset, val_dataset, test_dataset]
]


awd_model_name = 'awd_lstm_lm_1150'
awd_model, vocab = nlp.model.get_model(
    awd_model_name,
    vocab=vocab,
    dataset_name=dataset_name,
    pretrained=True,
    ctx=context[0])

print(awd_model)
print(vocab)


val_L = evaluate(awd_model, val_data, batch_size, context[0])
test_L = evaluate(awd_model, test_data, batch_size, context[0])

print('Best validation loss %.2f, val ppl %.2f' % (val_L, math.exp(val_L)))
print('Best test loss %.2f, test ppl %.2f' % (test_L, math.exp(test_L)))