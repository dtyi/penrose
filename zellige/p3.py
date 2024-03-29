from abc import ABC, abstractmethod

import numpy as np
from zellige.objmatrix import ObjMatrix


"""
For reference on what this code implements:
http://www.ams.org/publicoutreach/feature-column/fcarc-penrose
"""
DEBUG = False


class Edge:
    """
    A thin container class provided for convenience of adding tiles 
    to a growing patch, a la crystallization.
    
    This class is ancillary to P3, as the relations between rhombs 
    in the tiling system are fully defined by inflation/deflation. """
    def __init__(self, v0, v1, notches):
        self.v0 = v0
        self.v1 = v1
        self.notches = notches


class Deflation:
    """
    A Deflation has an inverse inflation.
    This class is generic to tiling systems, not P3 in particular, and
    may be refactored into a future generic tiling module.
    """
    def __init__(self, matrix, begin, end):
        """
        matrix: a 2x2 linear transform from vect of begin shape
            to the vect of a end (deflated, smaller) shape.

        begin, end: the types e.g. ThinA, FatA, of the parent polygon
            being deflated, and the result.
        """
        self.matrix = matrix
        self.inverse = matrix.inverse
        self.begin = begin
        self.end = end

    def install(self):
        
        self.begin.deflations += [(self.matrix,    self.end)]
        self.end.inflations += [(self.inverse, self.begin)]

    def __str__(self):
        return str(self.matrix)+str(self.begin)+str(self.end)


class BiDeflation(Deflation):
    """
    This subclass works with the left/right-handed P3 half-rhombs,
    which have symmetric inflations/deflations. This allows one
    definition to suffice instead of two (L/R) symmetric partners.
    This class could perhaps be better named DeflationWithShadow.
    """
    def shadow(self):
        return self.__class__(self.matrix.conj,
                              self.begin.shadow,
                              self.end.shadow)


class Polygon(object):
    """This class is generic to tiling systems, not P3 in particular.
    I just didn't make a generic tiling class
    to then inherit (subclass) from it."""

    def __init__(self, vect):
        """Vector assumes origin, ehat as a length-2 complex numpy
        column vector. Origin is an arbitrarily-chosen but consistent
        location in the polygon, e.g. a particular vertex,
        and ehat is an orientation/scale vector. """
        self.vect = vect.copy()
        """
        Strictly speaking, for an n-dimensional polytope, we need
        a n-dimensional vector location and an orientation + scale.
        It just happens to be that in 2d the rotation component
        of a vector suffices as orientation, and radial is scale.
        In 3d unfortunately SO(3) is 3-dimensional, and then you need
        another dimension for scale...
        """

    def __repr__(self):
        return type(self).__name__+"("+str(self.vect)+")"

    def __eq__(self, other):
        if type(self) == type(other):
            return all(self.vect == other.vect)
        return False

    def __hash__(self):
        return hash((self.vect[0], self.vect[1], type(self)))

    # The vertex locations (as column vect of complex numbers)
    # It's ok for this to be 1d, since numpy interprets it
    # as a column vector when performing matrix multiplication.
    @property
    def verts(self):
        return self.vert_transf @ self.vect

    # Vertex locations (nx2 array; n vertices * (x,y) coords)
    @property
    def real_verts(self):
        return np.stack((self.verts.real, self.verts.imag), axis=-1)

    # Vertex transformation:
    # This matrix maps the column vector [position, ehat]
    # to the vertex locations, and is nx2, for n vertices.
    # More generally, we just need some sort of function from
    # the polygon state vector to its vertices.
    @property
    def vert_transf(self):
        raise NotImplementedError

    # The direction the vertices wind, 1 for CCW, -1 for clockwise
    @property
    def handedness(self):
        raise NotImplementedError

    @property
    def edge_notches(self):
        raise NotImplementedError

    # Edges are ancillary to P3, as the relations between
    # rhombs in the tiling system are fully defined by the
    # inflation/deflation transforms. Provided for convenience
    # of adding tiles to a growing patch, a la crystallization.
    @property
    def edges(self):
        ret = []
        for v0, v1, n in zip(np.hstack((self.verts, self.verts[0:1])),
                             np.hstack((self.verts[1:], self.verts[0:2])),
                             self.edge_notches):
            ret.append(Edge(v0, v1, n))
        return ret

    def expand_edge(self, edge_num, desired_shapes=None):
        """Create a Polygon that matches an existing one 
        across a given edge.

        The prototypical use case is expanding a patch of tiles.

        Parameters
        ----------
        edge_num: int
            which edge; a 0-started index. (Clockwise or CCW depends on
            the handedness property.)
        desired_shapes: iterable of shape classes
            If None, uses the system attribute of this shape.

        Returns
        -------
        Polygon
        """
        # print(desired_shapes)
        # find which tiles match, and across which edge
        # notches_o: how many notches are on the given edge.
        # notches_t: target edge (o for own edge)
        # refactor:
        # for speed, matches can be cached after the polygons'
        # instantiation, instead of being re-found on every call.
        if desired_shapes is None:
            desired_shapes = self.system

        notches_o = self.edge_notches[edge_num]
        matches = []
        for shape in desired_shapes:
            for i, notches_t in enumerate(shape.edge_notches):
                if notches_t+notches_o == 0:
                    matches.append((shape, i))
        # print(matches)

        # print(edge_num-3,edge_num-1)
        # The vertices of the edge in question
        #edge = self.verts[origin_edge_index-3:origin_edge_index-1]
        # Fancy indexing allows us to take two consecutive points from verts,
        # wrapping around the end (slicing doesn't support this).
        edge = self.verts[np.arange(edge_num, edge_num+2) % 3]
        #edge = self.verts[origin_edge_index%3-3:(origin_edge_index+2)%3-3]
        ret = []
        for match in matches:
            # If we're matching a left-handed HalfRhomb to another
            # left-handed, or right to right, their rotation (rot)
            # will have the same signs. In that case, we have to
            # reverse the ordering of the vertices in the edge
            # in order to have them match the ordering in the new
            # HalfRhomb.
            # match[0] is the shape, [1] is the edge index
            # edge_transf is the matrix that transforms the defining
            # vect of the target polygon to the matched edge.
            # Since that edge is shared with the self Polygon
            # (up to rotation), the inverse transforms our edge to
            # the identity vector of the target shape.
            edge_transf = match[0].vert_transf[[match[1], (match[1]+1) % 3]]
            if self.handedness*match[0].handedness > 0:
                ret.append(match[0](edge_transf.inverse @ edge[::-1]))
            else:
                ret.append(match[0](edge_transf.inverse @ edge))
        return ret

    @property
    def system(self):
        """A polygon properly belongs in a aperiodic tiling system
        which should include the shapes that are related through
        inflation, deflation, or edge matching."""
        raise NotImplementedError

    # Deflations: a list of (matrix, polygontype) tuples.
    # matrix is a 2x2 linear transform from vect of begin shape
    # to the vect of a end (deflated, smaller) shape.
    # More generally, we need a multivalued function from this
    # polygon to child polygons.
    @property
    def deflations(self):
        raise NotImplementedError
    # Inflations are similar, except the mappings
    # (represented as matrices)
    # map from a smaller shape's [[center],[ehat]] to
    # the bigger, containing shape.
    @property
    def inflations(self):
        raise NotImplementedError

    
    def deflate(self):
        """
        Apply deflations; for every deflation in the list,
        instantiate a new Polygon on the matrix (dot) product
        of ``transformation_matrix @ location_orientation_vector``
        """
        return [fl[1](fl[0] @ self.vect) for fl in self.deflations]

    def inflate(self):
        return [fl[1](fl[0] @ self.vect) for fl in self.inflations]


# HalfRhomb and derived classes define the polygons
# that make up the P3 Penrose tiling.
# refactor:
# currently the patch-growing code works on halfRhombs, and
# performs a lot of extra work finding mirror halfrhombs.
# Maybe it's a good idea to add a Rhomb class and have it work on
# those.
class HalfRhomb(Polygon):
    # The basic rotation amount is pi/5
    incr = 1j*np.pi/5

    # Not filled in: rot (base angle, as multiple of incr),
    @property
    def rot(self):
        raise NotImplementedError

    @property
    def handedness(self):
        return 1-2*(self.rot < 0)  # sign, with 0 evaluating to 1

    # Return the corresponding other half-rhomb,
    # that completes the Penrose rhomb.
    @property
    def shadow(self):
        raise NotImplementedError

    def __neg__(self):
        return self.shadow(self.vect)


# Looking at the AMS column, the 'center' is the vertex
# in the rhombs at which the double arrows meet,
# while ehat is a vector with length equal to the
# edge length, aligned down the splitting line
# (line of symmetry.)
# In order for each class to have its own list object,
# not shared with the parent (super) and sibling classes,
# lists inflations and deflations cannot be inherited
# and have to be static attributes defined newly in each class.
class FatA(HalfRhomb):
    inflations, deflations = [], []
    rot = 1
    # Edge decorations are ancillary to Penrose tiling
    # system 3, but helpful for growing a patch a la
    # crystallization.
    edge_notches = [-2, -1, 4]


class FatB(HalfRhomb):
    inflations, deflations = [], []
    rot = -1
    edge_notches = [2, 1, -4]


# FatA and FatB together make a full Fat rhomb.
FatA.shadow = FatB
FatB.shadow = FatA


class ThinA(HalfRhomb):
    inflations, deflations = [], []
    rot = 2
    edge_notches = [-2, 1, 3]


class ThinB(HalfRhomb):
    inflations, deflations = [], []
    rot = -2
    edge_notches = [2, -1, -3]

    # def __neg__(self):
    #     return ThinA(self.vect)
    #kind,rot,width = 1,2,1/tau
    #kind,side = 1,1j*np.pi/5
    #vert_transf = np.array([[1,0],[1,np.exp(2*side)],[1,1/tau]])
ThinA.shadow = ThinB
ThinB.shadow = ThinA

CYCLOTOMIC = 0
FLOATING = 1


# if USE_PYTHON_FLOAT:
#     import float_implementation as implementation
# elif USE_FANCY_INTS:
#     import fancy_implementation as implementation

# A convenience grouping to distinguish the ground-level
# tiles from the superclasses in the class hierarchy
shapes = [FatA, FatB, ThinA, ThinB]


def make_halfrhombs(number_system=CYCLOTOMIC):
    # One list that contains all the shape classes that go together
    # system_shapes = []

    # system_shapes = [type(x.__name__, (x,), dict(x.__dict__)) 
    #     for x in shapes]

    for shape in shapes:
        #print(shape.rot, shape.incr)
        # print(width)

        # symbolic requires SumZ and ObjMatrix
        if number_system == CYCLOTOMIC:
            from zellige.sumz import SumZ, SumPol
            pol = SumPol
            shape.vert_transf = ObjMatrix(
                [[pol(0, 0), SumZ()],
                 [pol(0, 0), pol(shape.rot, 0)],
                 [pol(0, 0), pol(shape.rot, 0)+pol(-shape.rot, 0)]
                 ])
        else:
            width = 2*np.cosh(shape.rot*shape.incr)
            shape.vert_transf = np.array(
                [[1, 0],
                 [1, np.exp(shape.rot*shape.incr)],
                 [1, width]
                 ])

        shape.system = shapes

        shape.inflations = []
        shape.deflations = []

    # Specify the particular deflations that define the P3 tiling,
    # (along with the half-rhomb shapes).
    tau = 1./2+np.sqrt(5)/2
    if number_system == CYCLOTOMIC:
        from zellige.sumz import SumZ, SumPol
        pol = SumPol
        # requires ObjMatrix and SumZ
        deflations = [BiDeflation(ObjMatrix([[pol(0, 0), pol(0, -1)],
                                             [SumZ(), pol(3, -1)]]),
                                  ThinA, ThinA),
                      BiDeflation(ObjMatrix([[pol(0, 0), pol(2, 0)],
                                             [SumZ(), pol(-3, -1)]]),
                                  ThinA, FatA),
                      BiDeflation(ObjMatrix([[pol(0, 0), pol(1, 0)],
                                             [SumZ(), pol(-4, -1)]]),
                                  FatA, FatA),
                      BiDeflation(ObjMatrix([[pol(0, 0), pol(1, 0)],
                                             [SumZ(), pol(-1, -1)]]),
                                  FatA, ThinB),
                      BiDeflation(ObjMatrix([[pol(0, 0), pol(0, 1)],
                                             [SumZ(), pol(5, -1)]]),
                                  FatA, FatB),
                      ]
    else:
        deflations = [BiDeflation(ObjMatrix([[1, 1/tau],
                                             [0, np.exp(3j*np.pi/5)/tau]]),
                                  ThinA, ThinA),
                      BiDeflation(ObjMatrix([[1, np.exp(2j*np.pi/5)],
                                             [0, np.exp(-3j*np.pi/5)/tau]]),
                                  ThinA, FatA),
                      BiDeflation(ObjMatrix([[1, np.exp(1j*np.pi/5)],
                                             [0, np.exp(-4j*np.pi/5)/tau]]),
                                  FatA, FatA),
                      BiDeflation(ObjMatrix([[1, np.exp(1j*np.pi/5)],
                                             [0, np.exp(-1j*np.pi/5)/tau]]),
                                  FatA, ThinB),
                      BiDeflation(ObjMatrix([[1, tau],
                                             [0, np.exp(5j*np.pi/5)/tau]]),
                                  FatA, FatB),
                      ]

    # This bit of code works with the symmetric half-rhombs
    # in the P3 tiling, but the general idea of wiring up
    # the mappings between the (de)inflated polygon types
    # is generic to tiling systems.
    shadow_deflations = [deflation.shadow() for deflation in deflations]
    for deflation in deflations+shadow_deflations:
        deflation.install()

make_halfrhombs()