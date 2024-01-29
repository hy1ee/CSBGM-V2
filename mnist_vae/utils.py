


def print_hparams(hparams):
    print('')
    for temp in dir(hparams):
        if temp[:1] != '_':
            print('{0} = {1}'.format(temp, getattr(hparams, temp)))
    print('')