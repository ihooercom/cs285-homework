class BaseCritic(object):
    def update(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        raise NotImplementedError
