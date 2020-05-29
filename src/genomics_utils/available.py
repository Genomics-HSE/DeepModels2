import genomics_models
import genomics_data

__all__ = [
    'models',
    'experiments'
]

models = {
    attr: getattr(genomics_models, attr)
    for attr in genomics_models.__all__
}


experiments = {
    '{}'.format(model): models[model]
    for model in models
}