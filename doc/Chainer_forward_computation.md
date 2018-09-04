## FORWARD COMPUTATION

### Network Definition
Chainer's official MNIST example:
```python
class MyNetwork(Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        super(MyNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)

model = MyNetwork()
```

**Note the above code does not compatible with the latest version since they change the `__call__()` to `forward()`**

For our research, we change the network parameters and then check the influence. 

To perform forward computation and then check the result, one might write his own codes or use the methods offered by the framework. 
Here we use the `__call__` method the perform accuracy check.

What we care most is then the **x** paramter.

In Chainer's docs, I found this one:

```python
x = x[None, ...]
```
placehoder

Therefore, to perform forward computation in Chainer:
```python
x, t = test[0]
x = x[None, ...]
#print(x)
#print(t)
print(model.forward(x))
```
which gives:
```python
variable([[-16.477377  -6.184371 -20.764294 -18.087664 -11.697815
           -17.637484 -33.58191   27.882673 -19.956192  -7.090014]])
```
