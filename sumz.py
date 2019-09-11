from abc import ABC, abstractmethod
from functools import lru_cache
from scipy.linalg import circulant
import numpy as np

#refactor:
# This whole thing perhaps is best replaced with SymPy
from functools import total_ordering
@total_ordering # generates all 6 comparison magic methods from == and <
class Symb(ABC):
    """Symbolic sum of irrational terms, each with some integer
    coefficient."""
    @property
    @abstractmethod
    def terms(self):
        pass

    def __init__(self, coeffs):
        if any(np.rint(coeffs) != np.array(coeffs)):
            raise TypeError("Non-integer coefficients!")
        self.coeffs = np.array(coeffs)
        self.approx = sum(self.coeffs*self.terms)
    def __add__(self, other):
        return type(self)(self.coeffs+other.coeffs)
    def __sub__(self, other):
        return type(self)(self.coeffs-other.coeffs)
    def __neg__(self):
        return type(self)(-1*self.coeffs)
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            if (all(self.coeffs==0) and other==0):
                return True
            # hopefully a very conservative warning, since
            # floating-point error ~1e-16, tolerance here is 1e-8
            if __debug__:
                if np.isclose(self.approx, other):
                    raise ValueError("Uncertain floating-point"
                        " comparison",str(self),other)
            return self.approx == other
        # As long as the terms are all irrational ratios to each other,
        # it should be impossible for two of these to be equal with
        # non-equal coefficients.
        return all((self.coeffs-other.coeffs)==0)
    def __lt__(self, other): # whether self < other
        if not isinstance(other, type(self)):
            if (all(self.coeffs==0) and other==0):
                return False # in case of exact equality with zero
            if __debug__:
                if np.isclose(self.approx, other):
                    raise ValueError("Uncertain floating-point"
                        " comparison", str(self),other)
            # cmp() doesn't exist in Python3, so instead 
            # ((a > b) - (a < b))
            return self.approx < other
        otval = sum(other.coeffs*other.terms)
        return self.approx < otval
            
class RealSymb(Symb):
    terms = np.array([1,np.cos(72*np.pi/180),np.cos(36*np.pi/180)])
    def __str__(self):
        return "{} + {}*cos(72deg) + {}*cos(36deg) ~ {}".format(
            *self.coeffs,self.approx)
    
class ImagSymb(Symb):
    terms = np.array([np.sin(72*np.pi/180),np.sin(36*np.pi/180)])
    def __str__(self):
        return "{}*sin(72deg) + {}*sin(36deg) ~ {}".format(
            *self.coeffs,self.approx)
            

#@lru_cache(maxsize=1024)
class SumZ(object):
    """A representation of a complex number z
    as a sum of five equiangularly spaced unit basis vectors.
    Well, technically a spanning set, since they're linearly dependent.
    Well, actually, any four out of five are linearly independent
    as long as you're working in the integers (or rationals?)
    rather than reals, since no integral combination of three can 
    generate a fourth.
    
    This class can be dropped into the existing code to complex nums
    when initializing Polygons, within the vect argument.
    Unfortunately, for the sake of matrix multiplication,
    don't instantiate a Polygon with vect = np.array([[SumZ1],[SumZ2]]), 
    but ObjMatrix([[SumZ1],[SumZ2]]). This is bcause numpy doesn't
    support matrix multiplication with object arrays. See ObjMatrix
    for more details about numpy being wonky with arrays
    of objects and matrix multiplication."""
    #ehat=[np.exp(i*2j*np.pi/5) for i in range(5)]
    
    # a numpy array of 5 unit vectors, spaced at 72 degrees
    ehat = np.exp(np.arange(5)*2j*np.pi/5)
    tau = 1./2+np.sqrt(5)/2 # a numpy.float64
    
    # pm stands for Permutation Matrices for multiplication.
    #refactor:
    # Could be restated as circulant matrices.
    #pm = [np.roll(np.identity(5,dtype=np.int),i,axis=0) for i in range(5)]
    pm = [circulant(np.eye(5,dtype=np.int)[i]) for i in range(5)]
    
    rot_matrices=[None]*10
    rot_matrices[::2]=pm
    rot_matrices[1::2]=[-pm[3],-pm[4],-pm[0],-pm[1],-pm[2]]
    
    tau_matrix = circulant([0, 0, -1, -1, 0])
    inv_tau_matrix = circulant([0, 1, 0, 0, 1])
    
    
    def __init__(self, coeffs=[0]*5, *args):
        if len(args) > 0:
            self.coeffs = np.array((coeffs,)+args)
        else:
            self.coeffs=np.array(coeffs)
        # For convention, set the zeroth coefficient to 0
        # This makes it easier to check if two SumZ objects are equal.
        # Note that SumZ([0,0,0,0,0]) is the same point as 
        # SumZ([1,1,1,1,1])
        self.coeffs-=self.coeffs[0]

        if any(np.rint(self.coeffs) != np.array(self.coeffs)):
            raise ValueError("Non-integer coefficients!")
        self.coeffs = np.around(self.coeffs).astype(int)
        
    @classmethod
    def from_polar(cls, rotation, size):
        bigness = np.linalg.matrix_power(cls.tau_matrix,size)
        mat = bigness@cls.rot_matrices[rotation%10]
        return cls(mat[:,0])
        
    @property
    def coords(self):
        """Return my coordinates, as a complex number."""
        try:
            return self._coords
        except AttributeError:
            ret = 0
            # reorder addition in order to reduce real numerical error
            for coeff,eha in zip(self.coeffs[[0,1,4,2,3]], 
                                   self.ehat[[0,1,4,2,3]]):
                ret+=coeff*eha
            self._coords = ret
            # also, squash the imaginary part to 0 if the coefficients
            # are exactly balanced.
            if np.all(self.coeffs[[1,2]] == self.coeffs[[4,3]]):
                self._coords = self._coords.real+0j
            # ditto the real part
            if np.all(self.coeffs[[1,2]] == -1*self.coeffs[[4,3]]):
                self._coords = self._coords.imag*1j
        return self._coords
    
    @property
    def real(self):
        return self.coords.real

    @property
    def imag(self):
        return self.coords.imag
    
    
    @property
    def real_symb(self):
        # Symbolic representation of real part
        #return {1:self.coeffs[0],
        #        np.cos(72*np.pi/180):self.coeffs[1]+self.coeffs[4],
        #        np.cos(36*np.pi/180):-1*(self.coeffs[2]+self.coeffs[3])
        #       }
        return RealSymb([self.coeffs[0],
                         self.coeffs[1]+self.coeffs[4],
                         -1*(self.coeffs[2]+self.coeffs[3])
                        ])
    
    @property
    def imag_symb(self):
        # Symbolic representation of imaginary part
        #return {np.sin(72*np.pi/180):self.coeffs[1]-self.coeffs[4],
        #        np.sin(36*np.pi/180):self.coeffs[2]-self.coeffs[3]
        #       }
        return ImagSymb([self.coeffs[1]-self.coeffs[4],
                         self.coeffs[2]-self.coeffs[3] 
                        ])
    
    def conjugate(self):
        # basis vectors get sent like this:
        #from 0 1 2 3 4
        # to  0 4 3 2 1
        return SumZ(self.coeffs[[0,4,3,2,1]])
    
    @property
    def matrix(self):
        # multiplication by basis vector ehat0 is unity, ehat1 sends
        # all basis vectors to the next (e.g. coefficients
        # [0,1,2,3,4]->[4,0,1,2,3]), ehat2 sends them down by 2, etc. 
        # This can be seen as a permutation matrix.

        # more explicit form
        return sum(SumZ.pm[i]*self.coeffs[i] for i in range(5))
        # don't do this, since numpy broadcasting starts from
        # the trailing dimensions
        #return sum(pm*self.coeffs)
        
        #return circulant(self.coeffs) #alternative; might be faster
    
    def __add__(self, other):
        if other==0:
            return self
        # Only add to other instances of SumZ!
        if isinstance(other, SumZ):
            return SumZ(self.coeffs+other.coeffs)
        #raise TypeError("Only add to other instances of SumZ!")
        return NotImplemented
        
    
    def __sub__(self, other):
        if other==0:
            return self
        # Only add to other instances of SumZ!
        if isinstance(other, SumZ):
            return self+(-other)
        return NotImplemented
    
    def __neg__(self):
        return SumZ(-1*self.coeffs)
    
    def __mul__(self, other):
        if other==1:
            return self
        if other==0:
            return SumZ()

        # SumZ objects, mathematically speaking, are a subset of 
        # complex numbers. Multiplication by them is particularly nice.
        # the 5 basis vectors act on each other
        # with addition as on the group Z5
        if isinstance(other, SumZ):
            return SumZ(self.matrix @ other.coeffs)

        # If the other thing is iterable, broadcast over it.
        # What's the precedence of mul and rmul when both operands define it?
            
        # Multiplication really only works with complex nums,
        # of form tau**n*e**(m*i*pi/5)
        # IOW z=re^(i\theta), where r=\tau^n and \theta=m*pi/5
        # How much rotation, in increments of pi/5?
        # (10 is a complete circle)
        #if not isinstance(other, complex)
        # Unfortunately, isinstance(1.0, complex)==False,
        # where in math reals \subset complex.

        rot = int(np.rint(np.angle(other)*5/np.pi))
        #if rot < 0: # always rotate counterclockwise
        #    rot += 10
        #coeffs2 = self.coeffs
        #for i in range(rot):
        #    coeffs2 = -np.roll(coeffs2,3)
            
        # rolling/inverting: inverting is like rolling by 2.5 indices
        # (or 5 of 10 ticks) if number of ticks is odd, invert it. 
        # Add 5 to number of ticks.
        if rot%2 != 0:
            coeffs2 = -1* self.coeffs
            rot += 5
        else:
            coeffs2 = self.coeffs
        coeffs2 = np.roll(coeffs2, rot//2)
        
        # Avoid zero-logarithm errors by this shortcut on zero.
        #if DEBUG:
        #    if np.isclose(0,np.abs(other)):
        #        return SumZ([0]*5)
        if np.abs(other)<.00000001:
            return SumZ([0]*5)
        # How many inflation/deflation scaling steps?
        # Basically, what exponent of tau multiplies the scale?
        logr = np.log(np.abs(other))/np.log(self.tau)
        updown = int(np.rint(logr))
        if __debug__:
            if not np.isclose(logr,np.rint(logr)):
                raise ValueError("log_tau( {} )== {:.8f} provokes"
                    " large rounding".format(other,logr))
        if updown>0:
            for _ in range(updown):
                # (circularly) convolve the existing coefficients'
                # by sending e1 to -(e3+e4), e2 to -(e1+e3), etc.
                # e.g. e3 to -e0-e1
                new = np.zeros(5,dtype=np.int)
                for i in range(5):
                    # For each basis vect, replace with a lin sum
                    # of others
                    new[i-2]-=coeffs2[i]
                    new[i-3]-=coeffs2[i]
                coeffs2 = new
        elif updown<0:
            for i in range(-updown):
                # (circularly) convolve the existing coefficients'
                # by sending e1 to e0+e2, e2 to e1+e3, etc.
                # (alternative: e4 to -e1-e2-e4)
                new = np.zeros(5,dtype=np.int)
                for i in range(5):
                    # For each basis vect, replace with a lin sum
                    # of others
                    new[i-1]+=coeffs2[i]
                    new[i-4]+=coeffs2[i]
                coeffs2 = new
        return SumZ(coeffs2)

        
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        if isinstance(other, SumPol):
            return self*other.inv
        if isinstance(other, SumZ):
            # Is there a way to detect when one is a rational multiple
            # of the other? If possible, try to stay exact, and avoid
            # floating-point arithmetic.
            
            # Try to address the case when one is an exact multiple
            # of the other.
            # from fractions import Fraction
            # ratio = None
            # for i in range(5):
            #     # if there's a zero in one but not the other, they
            #     # can't possibly be multiples.
            #     if (self.coeffs[i]==0) != (other.coeffs[i]==0):
            #         break # go to fallback technique
            #     elif self.coeffs[i]==0 and other.coeffs[i]==0:
            #         continue
            #     else: 
            #         if ratio is None: # First nonzero coefficient pair
            #             ratio = Fraction(self.coeffs[i],other.coeffs[i])
            #         if ratio != Fraction(self.coeffs[i],other.coeffs[i]):
            #             break # if this pair's ratio isn't identical
            # else: # If no incompatibilities have been found.
            #     #print("returning exact ratio")
            #     return SumZ([ratio,0,0,0,0])
            
            try:
                # This is a hack to try to maintain floating-point 
                # safety. If we're really close, let it slide and round.
                # I'm not sure how it'd work on fractional lengths;
                # it's not really designed for that.
                newcoeffs = np.linalg.solve(other.matrix, self.coeffs)
                newcoeffs -= newcoeffs[0]
                if __debug__:
                    if not np.allclose(newcoeffs, np.rint(newcoeffs),
                        rtol=1e-10,atol=1e-10):
                        raise ValueError(str(newcoeffs)+" input"
                            " provokes large rounding, or FLOP error"
                            " exceeds 1e-10")
                return SumZ(np.rint(newcoeffs).astype(np.int))
            except np.linalg.LinAlgError: # When the other matrix is singular
                raise np.linalg.LinAlgError("Singular division",self, other)
                # print(self*other.conjugate,
                #     (other*other.conjugate).coords)
                # return ((self*other.conjugate)
                #     /(other*other.conjugate).coords)

        # For when dividing by an int, float, etc.
        return SumZ(self.coeffs/other)

    
    def __eq__(self, other):
        if isinstance(other, SumZ):
            return np.array_equal(self.coeffs, other.coeffs)
        else:
            return self.coords==other
    
    def __hash__(self):
        return hash(self.coeffs.tobytes())
    
    def __repr__(self):
        return "{}([{},{},{},{},{}])".format(self.__class__.__name__,
            *self.coeffs)
    
class SumPol(SumZ):
    def __init__(self, rotation=0, size=0):
        self.rotation = rotation%10
        self.size = size
        bigness = np.linalg.matrix_power(self.tau_matrix, size)
        #bigness -= bigness[0,0]
        mat = bigness @ self.rot_matrices[rotation%10]
        super().__init__(mat[:,0])

    def __neg__(self):
        return self.__class__(self.rotation+5,self.size)

    def __mul__(self, other):
        if other==1:
            return self
        
        #if not isinstance(other, self.__class__):
        #    raise TypeError("Only compatible with 1 and SumPol")
        if isinstance(other, self.__class__):
            return self.__class__(self.rotation+other.rotation,
                                  self.size+other.size)
        return super().__mul__(other)

    @property
    def inv(self):
        return self.__class__(-self.rotation, -self.size)

    def __rtruediv__(self, other):
        return other*self.inv

    #refactor: rewrite division as multiplicative inverse
    # def __truediv__(self, other):
    #     # self/other
    #     if other==1:
    #         return self
    #     else:
    #         if not isinstance(other, self.__class__):
    #             print(self,other)
    #             raise TypeError("Only compatible with 1 and SumPol"+str(self)+str(other))
    #         return self.__class__(self.rotation-other.rotation,
    #                               self.size-other.size)
    #     # Safe: Pol*Pol or Pol/Pol -> Polar
    #     # Unknown: Pol*SumZ or SumZ/Pol or Pol/SumZ -> ???
    #     # the safe thing to do would be to just return SumZ objects.
    #     # What we might want to do is automatically detect if the result
    #     # Could possibly be a SumPol.
    #     # There are 5 possibilities of orientation to check,
    #     # but how can we detect if the matrix is a power of the size 
    #     # (tau) matrix?
    # def __rtruediv__(self, other):
    #     # other/self
    #     if other==1:
    #         return self.__class__(-self.rotation, -self.size)
    #     else:
    #         if not isinstance(other, self.__class__):
    #             print(self,other)
    #             raise TypeError("Only compatible with 1 and SumPol"+str(self)+str(other))
    #         return self.__class__(other.rotation-self.rotation,
    #                               other.size-self.size)
        
    def __add__(self, other):
        # One obvious case is if the other number is zero
        if other==0:
            return self
        if isinstance(other, SumZ):
            # Zero case caught by if above.
            # if all(other.coeffs==np.zeros(5)):
            #     return self # then return a SumPol (self)
            # else: # Otherwise, fallback to SumZ
            return super().__add__(other)
        else:
            raise TypeError("Only add to other instances of SumZ!")
    
    
    
# check the numerical safety of pure-complex/pure-real
assert(SumZ((0,1,2,-2,-1)).coords.real==0)
assert(SumZ((0,1,2,2,1)).coords.imag==0)
_foo = SumZ(0,1,2,3,4)
_bar = SumZ((6,3,1,8,5))
_baz = SumZ((4, 1, 1, 7, 2))
# check commutativity of multiplication
assert(_foo*_bar==_bar*_foo)
# check associativity
assert(_foo*(_bar*_baz)==(_foo*_bar)*_baz)
# check distributivity
assert(_foo*(_bar+_baz)==_foo*_bar+_foo*_baz)
# Hooray, it's a commutative ring!

# multiplicative inverse (division) works?
#foo = SumZ(np.arange(5))
assert((_foo/_foo)==SumZ(1,0,0,0,0))
# So does a slightly nontrivial example
#foo = SumZ(np.arange(5))
#bar = SumZ([0,1,0,0,0])
assert((_foo*_bar)/_bar==_foo)
assert(np.isclose(((_foo*SumZ.tau**-2)/_foo).coords,SumZ.tau**-2))

ONE = SumPol(0,0)
ZERO = SumZ()
TAU = SumPol(0,1)