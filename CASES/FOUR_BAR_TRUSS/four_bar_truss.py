
def four_bar_truss(x):
    """

    This function evaluates the volume and displacement
    of a four-bar truss system.

    Parameters
    ----------
    x : array
        The input sample.

    Returns
    -------
    V : float
        The volume of the truss.
    d : float
        The displacement of the node.

    """

    V = x['L'] * (2. * x['A_1'] + 2.**(0.5) * x['A_2'] + x['A_3']**(0.5) + x['A_4'])

    d = x['F'] * x['L'] * (2. / (x['A_1'] * x['E_1']) +
                 2. * 2**(0.5) / (x['A_2'] * x['E_2']) -
                 2. * 2**(0.5) / (x['A_3'] * x['E_3']) +
                 2. / (x['A_4'] * x['E_4']))

    return V, d
