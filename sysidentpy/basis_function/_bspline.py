import numpy as np


class _BSplineWaveletsBase:
    """ A base class for all b-spline wavelet
    """
    pass


class MetaBSplineWavelets(type):
    """ A meta class to record ``MetaBSplineWavelets`` subclasses.
    """

    __slots__ = ()

    cls_map = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if issubclass(cls, _BSplineWaveletsBase):
            mcs.cls_map[namespace.get("order")] = cls
        return cls

    @classmethod
    def get(mcs, order):
        if order not in mcs.cls_map:
            raise NotImplementedError(
                "B-spline wavelet with order {} not implemented!".format(
                    order))
        return mcs.cls_map.get(order)


class BSplineWavelets(_BSplineWaveletsBase, metaclass=MetaBSplineWavelets):
    """ BSplineWavelets base class, to get order N b-spline wavelets instance.
    """

    order = None

    def phi(self, x):
        """ scaling function

        https://handwiki.org/wiki/Spline_wavelet#Cardinal_B-splines

        Args:
            x (_type_): input
        """
        raise NotImplementedError

    def psi(self, x):
        """ wavelet function

        https://handwiki.org/wiki/Spline_wavelet#Cardinal_B-splines

        Args:
            x (_type_): input
        """
        raise NotImplementedError

    def phi_j_k(self, x, j, k):
        return 2**(j / 2) * self.phi(2**j * x - k)

    def psi_j_k(self, x, j, k):
        return 2**(j / 2) * self.psi(2**j * x - k)

    @property
    def phi_range(self):
        phi_j = 0
        phi_range = range(-(self.order - 1), 1)
        return [(phi_j, k) for k in phi_range]

    @property
    def psi_range(self):
        psi_ranges = []
        for psi_j in range(self.order + 1):
            psi_range = range(-2**(self.order - 1) + 1, 2**psi_j)
            psi_ranges.extend([(psi_j, k) for k in psi_range])

        return psi_ranges

    @property
    def phi_len(self):
        return len(self.phi_range)

    @property
    def psi_len(self):
        return len(self.psi_range)

    @property
    def repetition(self):
        return self.phi_len + self.psi_len

    def decompose(self, x):
        return np.hstack(
            ([self.phi_j_k(x, j, k) for j, k in self.phi_range] +  # make phi
             [self.psi_j_k(x, j, k) for j, k in self.psi_range]))  # make psi

    @staticmethod
    def get_bspline(order, *args, **kw):
        cls = MetaBSplineWavelets.get(order)
        return (cls(*args, **kw) if cls else BSplineWavelets(*args, **kw))


class BSpline1(BSplineWavelets):
    """ b-spline wavelets order 1
    """
    order = 1

    def phi(self, x):
        return np.piecewise(x, [x < 0, ((0 <= x) & (x < 1)), x >= 1],
                            [0, 1, 0])

    def psi(self, x):
        return np.piecewise(x, [
            x < 0, ((0 <= x) & (x < 1 / 2)), ((1 / 2 <= x) & (x < 1)), x >= 1
        ], [0, 1, -1, 0])


class BSpline2(BSplineWavelets):
    """ b-spline wavelets order 2
    """
    order = 2

    def phi(self, x):
        return np.piecewise(
            x, [x < 0, ((0 <= x) & (x < 1)), ((1 <= x) & (x < 2)), x >= 2],
            [0, lambda x: x, lambda x: -x + 2, 0])

    def psi(self, x):
        return np.piecewise(x, [
            x < 0, ((0 <= x) & (x < 1 / 2)), ((1 / 2 <= x) & (x < 1)),
            ((1 <= x) & (x < 3 / 2)), ((3 / 2 <= x) & (x < 2)),
            ((2 <= x) & (x < 5 / 2)), ((5 / 2 <= x) & (x < 3)), x >= 3
        ], [
            0, lambda x: 1 / 6 * x, lambda x: -7 / 6 * x + 2 / 3,
            lambda x: 8 / 3 * x - 19 / 6, lambda x: -8 / 3 * x + 29 / 6,
            lambda x: 7 / 6 * x - 17 / 6, lambda x: -1 / 6 * x + 1 / 2, 0
        ])


class BSpline3(BSplineWavelets):
    """ b-spline wavelets order 3
    """
    order = 3

    def phi(self, x):
        return np.piecewise(x, [
            x < 0, ((0 <= x) & (x < 1)), ((1 <= x) & (x < 2)),
            ((2 <= x) & (x < 3)), x >= 3
        ], [
            0, lambda x: 1 / 2 * x**2, lambda x: -x**2 + 3 * x - 3 / 2,
            lambda x: 1 / 2 * x**2 - 3 * x + 9 / 2, 0
        ])

    def psi(self, x):
        return np.piecewise(x, [
            x < 0, ((0 <= x) & (x < 1 / 2)), ((1 / 2 <= x) & (x < 1)),
            ((1 <= x) & (x < 3 / 2)), ((3 / 2 <= x) & (x < 2)),
            ((2 <= x) & (x < 5 / 2)), ((5 / 2 <= x) & (x < 3)),
            ((3 <= x) & (x < 7 / 2)), ((7 / 2 <= x) & (x < 4)),
            ((4 <= x) & (x < 9 / 2)), ((9 / 2 <= x) & (x < 5)), x >= 5
        ], [
            0, lambda x: 1 / 240 * x**2,
            lambda x: -31 / 240 * x**2 + 2 / 15 * x - 1 / 30,
            lambda x: 103 / 120 * x**2 - 221 / 120 * x + 229 / 240,
            lambda x: -313 / 120 * x**2 + 1027 / 120 * x - 1643 / 240,
            lambda x: 22 / 5 * x**2 - 779 / 40 * x + 339 / 16,
            lambda x: -22 / 5 * x**2 + 981 / 40 * x - 541 / 16,
            lambda x: 313 / 120 * x**2 - 701 / 40 * x + 2341 / 80,
            lambda x: -103 / 120 * x**2 + 809 / 120 * x - 3169 / 240,
            lambda x: 31 / 240 * x**2 - 139 / 120 * x + 623 / 240,
            lambda x: -1 / 240 * x**2 + 1 / 24 * x - 5 / 48, 0
        ])


class BSpline4(BSplineWavelets):
    """ b-spline wavelets order 4
    """
    order = 4

    def phi(self, x):
        return np.piecewise(x, [
            x < 0, ((0 <= x) & (x < 1)), ((1 <= x) & (x < 2)),
            ((2 <= x) & (x < 3)), ((3 <= x) & (x < 4)), x >= 4
        ], [
            0, lambda x: 1 / 6 * x**3,
            lambda x: -1 / 2 * x**3 + 2 * x**2 - 2 * x + 2 / 3,
            lambda x: 1 / 2 * x**3 - 4 * x**2 + 10 * x - 22 / 3,
            lambda x: -1 / 6 * x**3 + 2 * x**2 - 8 * x + 32 / 3, 0
        ])

    def psi(self, x):
        return np.piecewise(x, [
            x < 0, ((0 <= x) & (x < 1 / 2)), ((1 / 2 <= x) & (x < 1)),
            ((1 <= x) & (x < 3 / 2)), ((3 / 2 <= x) & (x < 2)),
            ((2 <= x) & (x < 5 / 2)), ((5 / 2 <= x) & (x < 3)),
            ((3 <= x) & (x < 7 / 2)), ((7 / 2 <= x) & (x < 4)),
            ((4 <= x) & (x < 9 / 2)), ((9 / 2 <= x) & (x < 5)),
            ((5 <= x) & (x < 11 / 2)), ((11 / 2 <= x) & (x < 6)),
            ((6 <= x) & (x < 13 / 2)), ((13 / 2 <= x) & (x < 7)), x >= 7
        ], [
            0, lambda x: 1 / 30240 * x**3, lambda x: -127 / 30240 * x**3 + 2 /
            315 * x**2 - 1 / 315 * x + 1 / 1890, lambda x: 19 / 280 * x**3 - 47
            / 224 * x**2 + 2147 / 10080 * x - 103 / 1440, lambda x: -1109 /
            2520 * x**3 + 465 / 224 * x**2 - 32413 / 10080 * x + 16559 / 10080,
            lambda x: 5261 / 3360 * x**3 - 33463 / 3360 * x**2 + 42043 / 2016 *
            x - 145193 / 10080, lambda x: -35033 / 10080 * x**3 + 93577 / 3360
            * x**2 - 148517 / 2016 * x + 216269 / 3360, lambda x: 4832 / 945 *
            x**3 - 27691 / 560 * x**2 + 113923 / 720 * x - 28145 / 168,
            lambda x: -4832 / 945 * x**3 + 58393 / 1008 * x**2 - 52223 / 240 *
            x + 2048227 / 7560, lambda x: 35033 / 10080 * x**3 - 75827 / 1680 *
            x**2 + 981101 / 5040 * x - 234149 / 840, lambda x: -5261 / 3360 * x
            **3 + 38509 / 1680 * x**2 - 112487 / 1008 * x + 30347 / 168,
            lambda x: 1109 / 2520 * x**3 - 24077 / 3360 * x**2 + 78311 / 2016 *
            x - 141311 / 2016, lambda x: -19 / 280 * x**3 + 1361 / 1120 * x**2
            - 14617 / 2016 * x + 4151 / 288, lambda x: 127 / 30240 * x**3 - 55
            / 672 * x**2 + 5359 / 10080 * x - 11603 / 10080, lambda x: -1 /
            30240 * x**3 + 1 / 1440 * x**2 - 7 / 1440 * x + 49 / 4320, 0
        ])


class BSpline5(BSplineWavelets):
    """ b-spline wavelets order 5
    """
    order = 5

    def phi(self, x):
        return np.piecewise(x, [
            x < 0, ((0 <= x) & (x < 1)), ((1 <= x) & (x < 2)),
            ((2 <= x) & (x < 3)), ((3 <= x) & (x < 4)),
            ((4 <= x) & (x < 5)), x >= 5
        ], [
            0, lambda x: 1 / 24 * x**4, lambda x: -1 / 6 * x**4 + 5 / 6 * x**3
            - 5 / 4 * x**2 + 5 / 6 * x - 5 / 24, lambda x: 1 / 4 * x**4 - 5 / 2
            * x**3 + 35 / 4 * x**2 - 25 / 2 * x + 155 / 24, lambda x: -1 / 6 *
            x**4 + 5 / 2 * x**3 - 55 / 4 * x**2 + 65 / 2 * x - 655 / 24,
            lambda x: 1 / 24 * x**4 - 5 / 6 * x**3 + 25 / 4 * x**2 - 125 / 6 *
            x + 625 / 24, 0
        ])

    def psi(self, x):
        return np.piecewise(
            x,
            [
                x < 0, ((0 <= x) & (x < 1 / 2)), ((1 / 2 <= x) & (x < 1)),
                ((1 <= x) & (x < 3 / 2)), ((3 / 2 <= x) & (x < 2)),
                ((2 <= x) & (x < 5 / 2)), ((5 / 2 <= x) & (x < 3)),
                ((3 <= x) & (x < 7 / 2)), ((7 / 2 <= x) & (x < 4)),
                ((4 <= x) & (x < 9 / 2)), ((9 / 2 <= x) & (x < 5)),
                ((5 <= x) & (x < 11 / 2)), ((11 / 2 <= x) & (x < 6)),
                ((6 <= x) & (x < 13 / 2)), ((13 / 2 <= x) & (x < 7)),
                ((7 <= x) & (x < 15 / 2)), ((15 / 2 <= x) & (x < 8)),
                ((8 <= x) & (x < 17 / 2)), ((17 / 2 <= x) & (x < 9)), x >= 9
            ],
            [
                0,
                lambda x: 1 / 8709120 * x**4,
                lambda x: -73 / 1244160 * x**4 + 1 / 8505 * x**3 - 1 / 11340 *
                x**2 + 1 / 34020 * x - 1 / 272160,
                lambda x: 9581 / 4354560 * x**4 - 19417 / 2177280 * x**3 + 1303
                / 96768 * x**2 - 19609 / 2177280 * x + 6547 / 2903040,
                lambda x: -118931 / 4354560 * x**4 + 366119 / 2177280 * x**3 -
                186253 / 483840 * x**2 + 121121 / 311040 * x - 427181 /
                2903040,
                lambda x: 759239 / 4354560 * x**4 - 3146561 / 2177280 * x**3 +
                6466601 / 1451520 * x**2 - 13202873 / 2177280 * x + 26819897 /
                8709120,
                lambda x: -2980409 / 4354560 * x**4 + 5183893 / 725760 * x**3 -
                13426333 / 483840 * x**2 + 426589 / 8960 * x - 12635243 /
                414720,
                lambda x: 7873577 / 4354560 * x**4 - 16524079 / 725760 * x**3 +
                7385369 / 69120 * x**2 - 17868671 / 80640 * x + 497668543 /
                2903040,  # correct 2903040 not 290304
                lambda x: -14714327 / 4354560 * x**4 + 108543091 / 2177280 * x
                **3 - 56901557 / 207360 * x**2 + 1454458651 / 2177280 * x -
                5286189059 / 8709120,
                lambda x: 15619 / 3402 * x**4 - 33822017 / 435456 * x**3 +
                15828929 / 32256 * x**2 - 597598433 / 435456 * x + 277413649 /
                193536,
                lambda x: -15619 / 3402 * x**4 + 38150335 / 435456 * x**3 -
                20157247 / 32256 * x**2 + 859841695 / 435456 * x - 64472345 /
                27648,
                lambda x: 14714327 / 4354560 * x**4 - 4466137 / 62208 * x**3 +
                165651247 / 290304 * x**2 - 875490655 / 435456 * x + 4614904015
                / 1741824,
                lambda x: -7873577 / 4354560 * x**4 + 30717383 / 725760 * x**3
                - 179437319 / 483840 * x**2 + 16606729 / 11520 * x - 869722273
                / 414720,
                lambda x: 2980409 / 4354560 * x**4 - 12698561 / 725760 * x**3 +
                16211669 / 96768 * x**2 - 19138891 / 26880 * x + 3289787993 /
                2903040,
                lambda x: -759239 / 4354560 * x**4 + 10519741 / 2177280 * x**3
                - 10403603 / 207360 * x**2 + 71964499 / 311040 * x - 3481646837
                / 8709120,
                lambda x: 118931 / 4354560 * x**4 - 1774639 / 2177280 * x**3 +
                630259 / 69120 * x**2 - 14096161 / 311040 * x + 245108501 /
                2903040,
                lambda x: -9581 / 4354560 * x**4 + 21863 / 311040 * x**3 -
                407387 / 483840 * x**2 + 9758873 / 2177280 * x - 25971499 /
                2903040,
                lambda x: 73 / 1244160 * x**4 - 4343 / 2177280 * x**3 + 5273 /
                207360 * x**2 - 313703 / 2177280 * x + 380873 / 1244160,
                lambda x: -1 / 8709120 * x**4 + 1 / 241920 * x**3 - 1 / 17920 *
                x**2 + 3 / 8960 * x - 27 / 35840,
                0
            ])
