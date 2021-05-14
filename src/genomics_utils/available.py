import small_models
import full_models

__all__ = [
    'models',
    'experiments'
]

models = {
    attr: getattr(full_models, attr)
    for attr in full_models.__all__
}

for attr in small_models.__all__:
    models[attr] = getattr(small_models, attr)


experiments = {
    '{}'.format(model): models[model]
    for model in models
}
