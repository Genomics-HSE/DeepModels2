import squeeze_models
import sequence_models

__all__ = [
    'models',
    'experiments'
]

models = {
    attr: getattr(sequence_models, attr)
    for attr in sequence_models.__all__
}


experiments = {
    '{}'.format(model): models[model]
    for model in models
}
