import mxnet as mx
from mxnet import np, npx 
def test_rnn():
    #mx.random.seed(123)
    INT_OVERFLOW = 2**10
    state_size = 1
    def batch_check(x, modes, params):
        for m, p in zip(modes, params):
            state = np.random.normal(0, 1, (1, 4, state_size))
            x.attach_grad()
            state.attach_grad()
            x.attach_grad()
            p.attach_grad()

            with mx.autograd.record():
                y = npx.rnn(data=x, parameters=p, mode=m, \
                    state=state, state_size=state_size, num_layers=1)
            assert y.shape == (INT_OVERFLOW, 4, state_size)
            assert type(y[0]).__name__ == 'ndarray'
            y.backward()
            #print(y)
            #print(y)
            #mx.nd.waitall()
            print(state.grad)
    data = np.random.normal(0, 1, (INT_OVERFLOW, 4, 4))
    modes = ['rnn_relu',
             'rnn_tanh',
              'gru'
            ] #/// (input_size + state_size + 2) * size
    params = [np.random.normal(0, 1, ((4 + state_size + 2)*state_size),), \
         np.random.normal(0, 1, ((4 + state_size + 2)*state_size),), \
         np.random.normal(0, 1, ((4 + state_size + 2)*state_size*3),) \
         ]
    batch_check(data, modes, params)  




def test_rnn2():
    #mx.random.seed(123)
    INT_OVERFLOW = 2**5
    state_size = 2
    def batch_check(x, modes, params):
        for m, p in zip(modes, params):
            state = np.random.normal(0, 1, (1, 4, state_size))
            x.attach_grad()
            state.attach_grad()
            x.attach_grad()
            p.attach_grad()

            with mx.autograd.record():
                y = npx.rnn(data=x, parameters=p, mode=m, \
                    state=state, state_size=state_size, num_layers=1)
            assert y.shape == (INT_OVERFLOW, 4, state_size)
            assert type(y[0]).__name__ == 'ndarray'
            y.backward()
            #print(y)
            #print(y)
            #mx.nd.waitall()
            print(state.grad)
    data = np.random.normal(0, 1, (INT_OVERFLOW, 4, 4))
    modes = ['rnn_relu',
             'rnn_tanh',
              'gru'
            ] #/// (input_size + state_size + 2) * size
    params = [np.random.normal(0, 1, ((4 + state_size + 2)*state_size),), \
         np.random.normal(0, 1, ((4 + state_size + 2)*state_size),), \
         np.random.normal(0, 1, ((4 + state_size + 2)*state_size*3),) \
         ]
    batch_check(data, modes, params)  
test_rnn2()



