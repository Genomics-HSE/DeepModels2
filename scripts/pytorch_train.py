import os
import torch
import tqdm

from genomics import Classifier
from genomics_utils import ensure_directories


def train(dataset, model, device, seed, n_epochs, batch_size,
          output_path, logger, num_workers=1, quiet=False):
    model_root, = ensure_directories(output_path, 'models/')
    parameters_path = os.path.join(
        model_root,
        '{model}.pt'.format(model=model.name)
    )
    device = torch.device(device)
    torch.manual_seed(seed)
    
    clf = Classifier(model, logger=logger, device=device)
    if not quiet:
        print("Running {}-model...".format(model.name))
    losses = clf.fit(dataset, n_epochs=n_epochs, progress=None if quiet else tqdm)
    
    clf.save(parameters_path, quiet)
    
    logger.log_losses("dataset", model.name, losses)


def test(dataset, model, device, batch_size,
         output_path, logger, project, workspace):
    model_root, = ensure_directories(output_path, 'models/')
    parameters_path = os.path.join(
        model_root,
        '{model}.pt'.format(model=model.name)
    )
    device = torch.device(device)
    
    # train_loader = torch.utils.data.DataLoader(dataset.train_set, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(dataset.test_set, batch_size=batch_size, shuffle=True)
    
    clf = Classifier(model, logger=logger, device=device)
    
    state_dict = torch.load(parameters_path, map_location=torch.device(device))
    clf.classifier.load_state_dict(state_dict)
    
    # predictions_train, true_train = clf.predict(train_loader)
    # predictions_test, true_test = clf.predict(test_loader)
    #
    # accuracy_train = np.mean(np.argmax(predictions_train, axis=1) == true_train)
    # accuracy_test = np.mean(np.argmax(predictions_test, axis=1) == true_test)
    
    # todo
    heatmap_preds = clf.predict_proba(dataset, logger)
    
    # logger.log_metrics("d", model.name, accuracy_train=accuracy_train, accuracy_test=accuracy_test)
    logger.log_coalescent_heatmap(model.name, heatmap_preds, "00000")


class A:
    def a(self):
        print("a")


class B:
    def b(self):
        print("b")


class C(B, A):
    pass


if __name__ == '__main__':
    c = C()
    c.a()
    c.b()
