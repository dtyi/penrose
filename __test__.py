from sumz import ONE,ZERO,TAU
from objmatrix import ObjMatrix
foo = ObjMatrix([[ONE,ZERO],[ZERO,TAU]])
foo.__mul__(TAU)
foo*TAU
(ONE*TAU-ZERO*ZERO)*foo
foo.inverse