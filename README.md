# grokking
reproducing https://arxiv.org/pdf/2201.02177
also tested https://github.com/ironjr/grokfast

TODOs
* clean up train script
* increase batch size
* try more fns. make new ones, and impl S5 and mod div fns
* larger train sets converging weirdly?
* grokfast loop is pretty inefficient, surely there's a better way?
* grokfast not working, either set up wrong or need to configure better
* is torch.nn.utils.clip_grad_norm_ standard? try removing