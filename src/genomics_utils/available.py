import squeeze_models

__all__ = [
    'models',
    'experiments'
]

models = {
    attr: getattr(squeeze_models, attr)
    for attr in squeeze_models.__all__
}


experiments = {
    '{}'.format(model): models[model]
    for model in models
}
