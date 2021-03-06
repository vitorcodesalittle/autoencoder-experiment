import torch
import matplotlib.pyplot as plt

def evaluate_classifier(model, dataloader, debug=False):
  acc = 0
  for (x, ytrue) in dataloader:
    y = torch.argmax(model(x), 1)
    acc += torch.sum(y == ytrue)
  acc = acc / dataloader.dataset.data.shape[0]
  if debug:
    info = f"""Results - (tests)
    Accuracy: {acc.item():.3f}
    """
    print(info)
  return {'accuracy': acc.item()}


def plot_training_learning(history, filterkeys=[], path=None):
  for key in history:
    if key in filterkeys:
      continue
    plt.plot(
      history[key],
    )
  plt.legend(history.keys())
  if path:
      plt.savefig(path)
  else:
      plt.show()

def evaluate_autoencoder(model, testloader, metrics):
    results = [ [] for m in metrics ]
    for index, (x, _) in enumerate(testloader):
        y = model(x)
        for metricidx, metric in enumerate(metrics):
            results[metricidx] = metric(y, x)
    return results

def tensor_to_image(tensor):
    return tensor.detach().numpy().T

def visualize_autoencoded(model, xs):
    fig, axs = plt.subplots(len(xs), 2)
    for index, x in enumerate(xs):
        y = model(x)
        axs[index, 0].imshow(tensor_to_image(x))
        axs[index, 1].imshow(tensor_to_image(y))
    plt.show()
        
        


