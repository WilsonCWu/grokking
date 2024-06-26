# grokking
reproducing https://arxiv.org/pdf/2201.02177
also tested https://github.com/ironjr/grokfast

TODOs
* clean up train script, flag-guard grokfast, fix log paths so we can do several runs. should repro figure 1.
* increase batch size
* try more fns. make new ones, and impl S5 and mod div fns
* grokfast loop is pretty inefficient, surely there's a better way?
* grokfast was only marginally faster, figure out why. tune hyperparams? setup wrong?
* is torch.nn.utils.clip_grad_norm_ standard? try removing