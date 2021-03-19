
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

    L, F, E_1, E_2, E_3, E_4, A_1, A_2, A_3, A_4 = x

    V = L * (2. * A_1 + 2.**(0.5) * A_2 + A_3**(0.5) + A_4)

    d = F * L * (2. / (A_1 * E_1) +
                 2. * 2**(0.5) / (A_2 * E_2) -
                 2. * 2**(0.5) / (A_3 * E_3) +
                 2. / (A_4 * E_4))

    return V, d
