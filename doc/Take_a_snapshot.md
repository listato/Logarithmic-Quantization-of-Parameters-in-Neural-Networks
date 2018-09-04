```python
parser.add_argument('--resume', '-r', default='result\snapshot_iter_12000',
                        help='Resume the training from snapshot')
```

```python
# Take a snapshot for each specified epoch
frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
```

```python
if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)
```
