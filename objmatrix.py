import numpy as np

class ObjMatrix(np.ndarray):
    """
    For some reason numpy doesn't yet support matmul
    for object arrays. np.dot works fine though,
    simply devolving to standard python operators
    looped over the elements of the arrays.
    This class aims simply to patch in __matmul__
    (matrix multiplication) using np.dot.
    https://stackoverflow.com/questions/49386827/matrix-multiplication-with-object-arrays-in-python
    """
    def __matmul__(self, other):
        return ObjMatrix(np.dot(self, other))
    def __rmatmul__(self, other):
        return ObjMatrix(np.dot(other, self))

    def __mul__(self, other):
        # What's the precedence of mul and rmul when both operands define it?
        # Perhaps I could intentionally cede control when other is iterable.
        return np.vectorize(other.__mul__)(other)
    
    def __rmul__(self, other):
        return np.vectorize(other.__rmul__)(other)

    def __add__(self, other):
        return np.vectorize(other.__add__)(other)

    @property
    def real(self):
        """ndarray.real and ndarray.imag break when
        the array dtype is 'object', even if the individual
        objects implement object.real and object.imag,
        which is enough for np.real() and np.imag() to work
        on individual objects. This patches it in for
        arrays by returning obj.imag for every object 
        in the array, returning as usual an array of same shape."""
        return np.vectorize(lambda x: x.real)(self)
    @property
    def imag(self):
        return np.vectorize(lambda x: x.imag)(self)
    @property
    def conj(self):
        return np.vectorize(lambda x: x.conjugate)(self)
    #refactor (todo): rewrite this using __getattr__ in order
    # to be more general and truly an ObjMatrix rather than
    # just this thing I bodged together ad-hoc.
    
    #    return ObjMatrix([thing.imag for thing in self])
    # code below copied from the InfoArray example in docs
    # A bit cargo-culty; sorry.
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        #obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # I honestly don't know if this method is necessary.
        # this is a bit cargo-cultish, but the documentation
        # I skimmed seemed to imply this should be defined.
        
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        #self.info = getattr(obj, 'info', None)

    @property
    def inverse(self):
        if self.shape == (2,2):
            #Unpacking works without the tuple(), i.e. as a numpy array,
            # but pylint complains.
            a,b,c,d = tuple(self.flat)
            return self.__class__([[d,-b]
                                  ,[-c,a]])*(a*d-b*c)
        else:
            # pylint complains, but self.shape should be a tuple
            # (from numpy ndarrays) so it should be subscriptable.
            # print(type(self.shape))
            # if self.ndim==2 and self.shape[0]==self.shape[1]:
            #     raise NotImplementedError("Only 2x2 inverses implemented")
            # else:
            #     return np.linalg.LinAlgError("Non-square matrix")

            return np.linalg.inv(self)
        
        
# test_ehat = ObjMatrix([[SumZ([0,0,0,0,0])],
#                        [SumZ([1,0,0,0,0])]])
# matrix = np.array([[1,np.exp(2j*np.pi/5)],           
#                    [0,np.exp(-3j*np.pi/5)]])
# (matrix@test_ehat)