from tqdm import tqdm
import numpy as np



def iLQR(env, iterations=100,
         reg = 1,                  # initial regularization
         reg_lims = (1e-6, 1e10),  # limtis on regularization
         dreg = 2,                 # reg is multiplied or divided by dreg
         dreg_factor = 1.6,        # dreg is multiplied or diveded by dreg_factor):
         alpha = 1,
         initial_action = 0,       # initial actions randomized between (-initial_action, initial_action)
         verbose = False,
        ):

    # initial trajectory
    env.reset(reset_target=False)
    a = initial_action
    actions = [np.random.uniform([-initial_action]*env.action_dim, [initial_action]*env.action_dim)
               for i in range(env.max_steps)]
    states, costs, costs_derivs = env.rollout(actions)
    state_derivs = [env.state_derivs(s,a) for s,a in zip(states, actions)]
    history = dict(cost=[sum(costs)], reg=[reg])
    flag = ''

    # print column headers
    if verbose:
        headers = '{:12.12}{:12.12}{:12.12}{:12.12}{:12.12}{:12.12}{:12.12}'.format(
            'iteration', 'cost', 'dcost', 'expected', 'reg', 'status', 'max_action')
        print(headers, '\n' + ''.join(['-']*len(headers)))

    # optimize!
    for i in tqdm(range(iterations)) if not verbose else range(iterations):

        # differentiate trajectory
        # (only recompute if actions have changed since last iteration, i.e. if flat=='decreased')
        if flag=='decreased':
            states, costs, costs_derivs = env.rollout(actions)
            state_derivs = [env.state_derivs(s,a) for s,a in zip(states, actions)]

        # backward pass
        # (compute control modifications k and K)
        complete = False
        while not complete:
            k, K = [], []
            dV = 0
            V_x  = costs_derivs[-1]['l_x']
            V_xx = costs_derivs[-1]['l_xx']

            for t in range(env.max_steps-1, -1, -1):

                # quadratic cost approximation coefficients
                l, f = costs_derivs[t], state_derivs[t]
                Q_x  = l['l_x']  + f['f_x'].T @ V_x
                Q_u  = l['l_u']  + f['f_u'].T @ V_x
                Q_xx = l['l_xx'] + f['f_x'].T @ V_xx @ f['f_x']
                Q_uu = l['l_uu'] + f['f_u'].T @ V_xx @ f['f_u']
                Q_ux = l['l_ux'] + f['f_u'].T @ V_xx @ f['f_x']

                # compute controls k and K
                Q_uu = .5 * (Q_uu + Q_uu.T)  # make sure perfectely symmetric
                Q_uu_reg = Q_uu + np.diag(np.repeat(reg, len(Q_u)))

                try:
                    # cholesky decomposition instead of matrix inverse
                    L = np.linalg.cholesky(Q_uu_reg)
                    k.append(-np.linalg.solve(L.T, np.linalg.solve(L, Q_u)))
                    K.append(-np.linalg.solve(L.T, np.linalg.solve(L, Q_ux)))
                except np.linalg.LinAlgError:
                    # increase regularization if Q_uu non-positive definite
                    dreg = max(dreg_factor, dreg_factor*dreg)
                    reg = max(reg*dreg, reg_lims[0])
                    complete = False
                    break

                # update V
                dV += alpha**2*.5 * k[-1].T @ Q_uu @ k[-1] + alpha*k[-1].T @ Q_u  # expected cost reduction
                V_x  = Q_x  + K[-1].T @ Q_uu @ k[-1] + K[-1].T @ Q_u  + Q_ux.T @ k[-1]
                V_xx = Q_xx + K[-1].T @ Q_uu @ K[-1] + K[-1].T @ Q_ux + Q_ux.T @ K[-1]
                V_xx = .5 * (V_xx + V_xx.T)  # make sure perfectely symmetric
                complete = True

        k.reverse()
        K.reverse()

        # forward pass
        # (compute new trajectory with control modifications k and K)
        costs_new, actions_new = [], []
        env.reset(reset_target=False)

        for t in range(env.max_steps):
            actions_new.append(actions[t] + alpha*k[t] + K[t] @ (env.state - states[t]))
            costs_new.append(env.cost(env.state, actions_new[-1]))
            env.step(actions_new[-1])
        costs_new.append(env.cost_final(env.state))

        history['cost'].append(sum(costs_new))
        delta_cost = sum(costs_new) - sum(costs)

        # decide whether to keep new actions

        # increase regularization if cost increased
        if delta_cost>=0:
            dreg = max(dreg_factor, dreg_factor*dreg)
            reg = max(reg*dreg, reg_lims[0])
            flag = 'increased'

        # decrease regularization and update actions if cost decreased
        else:
            dreg = min(1/dreg_factor, dreg/dreg_factor)
            reg *= dreg * (reg > reg_lims[0])  # latter term sets reg=0 if reg<=reg_lims[0]
            flag = 'decreased'
            actions = actions_new.copy()

        history['reg'].append(reg)

        if verbose:
            print('{:6d}/{:<5d}{:<12.6f}{:<+12.2e}{:<+12.2e}{:<12.2e}{:<12s}{:<12.6f}'.format(
                i+1, iterations, history['cost'][-1], delta_cost, dV, reg, flag, np.abs(np.array(actions)).max()))

        if reg>reg_lims[1]:
            print('Regularization limit exceeded. Abandoning optimization.')
            break

    return actions, history
