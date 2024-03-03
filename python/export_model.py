import torch
import sys
def main(path):
    """Create a very simple ONNX model"""
    model = torch.nn.Linear(3, 2)

    w, b = model.state_dict()['weight'], model.state_dict()['bias']
    # w, b = model.weight, model.bias
    with torch.no_grad():
        w.copy_(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        b.copy_(torch.tensor([-1., -2.]))
    model.cuda()
    print('w = ', w, w.dtype)
    print('b = ', b, b.dtype)

    # Check the standard result, should be [1.5, 3.5]
    x = torch.tensor([0.5, -0.5, 1.0], device='cuda')
    y = model(x)
    print('x =', x)
    print('y =', y)

    # Export to ONNX
    model.eval()
    x = torch.randn(1, 3, requires_grad=True, device='cuda')
    
    #Export model.onnx with batch_size=1 
    print('\nExporting model.onnx ...')
    torch.onnx.export(model,
                      x,
                      path,
                      opset_version=9,
                      verbose=True,
                      export_params=True,
                      input_names=['input'],
                      output_names=['output'])
    
if __name__ == '__main__':
    main(sys.argv[1])