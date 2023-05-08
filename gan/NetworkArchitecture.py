topological_maps = ['floormap', 'wallmap', 'heightmap']

WGAN_gen = [
    {'n_filters': 1024, 'kernel_size': (4, 4), 'stride': (2, 2), 'dropout': True},
    {'n_filters': 512, 'kernel_size': (4, 4), 'stride': (2, 2), 'dropout': True},
    {'n_filters': 256, 'kernel_size': (4, 4), 'stride': (2, 2), 'dropout': False},
    {'n_filters': 128, 'kernel_size': (4, 4), 'stride': (2, 2), 'dropout': False}
]

WGAN_disc = [
    {'n_filters': 128, 'kernel_size': (4, 4), 'stride': (2, 2), 'norm': False},
    {'n_filters': 256, 'kernel_size': (4, 4), 'stride': (2, 2), 'norm': True},
    {'n_filters': 512, 'kernel_size': (4, 4), 'stride': (2, 2), 'norm': True},
    {'n_filters': 1024, 'kernel_size': (4, 4), 'stride': (2, 2), 'norm': True},
    ]

object_maps = ['monsters','ammunitions','powerups','artifacts','weapons']

CAN_gen = {'downstack': [
    {'n_filters': 128, 'kernel_size': (8, 8), 'stride': (4, 4), 'norm': False},
    {'n_filters': 256, 'kernel_size': (8, 8), 'stride': (4, 4), 'norm': True},
    {'n_filters': 512, 'kernel_size': (8, 8), 'stride': (4, 4), 'norm': True},
    {'n_filters': 1024, 'kernel_size': (8, 8), 'stride': (4, 4), 'norm': True},
  ],  'upstack': [
    {'n_filters': 1024, 'kernel_size': (8, 8), 'stride': (4, 4), 'dropout': True},
    {'n_filters': 512, 'kernel_size': (8, 8), 'stride': (4, 4), 'dropout': True},
    {'n_filters': 256, 'kernel_size': (8, 8), 'stride': (4, 4), 'dropout': True},
  ]}

CAN_disc = [
    {'n_filters': 128, 'kernel_size': (8, 8), 'stride': (4, 4), 'batchnorm': False},
    {'n_filters': 256, 'kernel_size': (8, 8), 'stride': (4, 4), 'batchnorm': True},
    {'n_filters': 512, 'kernel_size': (4, 4), 'stride': (2, 2), 'batchnorm': True},
    {'n_filters': 1024, 'kernel_size': (4, 4), 'stride': (2, 2), 'batchnorm': True},
]