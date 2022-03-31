import torch

def evaluate_classifier(model, dataloader, debug=False):
  acc = 0
  for i, (x, ytrue) in enumerate(dataloader):
    y = torch.argmax(model(x), 1)
    acc += torch.sum(y == ytrue)
  acc = acc / dataloader.dataset.data.shape[0]
  if debug:
    info = f"""Results - (tests)
    Accuracy: {acc.item():.3f}
    """
    print(info)
  return {'accuracy': acc.item()}


