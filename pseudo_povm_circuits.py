# -*- coding: utf-8 -*-
"""Contains classes QubitCircuit and PseudoPovm
Also contains some functions required for methods
of these classes
"""

import tensorflow as tf
import numpy as np

#Initialize pauli matrices
sigma_0 = tf.eye(2,dtype=tf.complex128)
sigma_x = tf.constant([[0,1],[1,0]],dtype = tf.complex128)
sigma_y = tf.constant([[0,-1j],[1j,0]],dtype = tf.complex128)
sigma_z = tf.constant([[1,0],[0,-1]],dtype = tf.complex128)

#All pauli in one (4,2,2) tensor
sigma = tf.concat([sigma_0[tf.newaxis],
                   sigma_x[tf.newaxis],
                   sigma_y[tf.newaxis],
                   sigma_z[tf.newaxis]],axis = 0)

class PseudoPovm:
  """Class of qubit pseudo POVM. Every matrix in pseudo MIC POVM is hermitian.
  In case of qubit we have 4 matrixes.
  Every matrix is a linear combination of pauli matrixes.
  Every matrix can be parametrised by 4 real numbers.
  We parametrise 3 of 4 matrixes.
  The last one is {E_3 =I - E_0 - E_1 - E2}
  So parametrisation has 4 * 3 = 12 numbers

  Attributes

  lam : tensor [4,3] of float64
    oarametrisation of pseudo POVM

  Methods

  get_lam()
    returns tensor of parametrisation of pseudo POVM

  get_pseudo_povm_in_comp_basis()
    converts parametrisation to 4 hermitian
    matrixes as [4,2,2] tensor of complex128

  measurement()
    initializes projective measurement
    matrix to computational basis

  init_state()
    initializes vector of quantum state |0> in given pseudo POVM
  """

  def __init__(self, lam):
    self.lam = lam

  def get_lam(self):
    """Returns parametrisation of pseudo POVM as [4,3] tensor"""

    return self.lam

  def get_pseudo_povm_in_comp_basis(self):
    """Converts parametrisation to 4 hermitian matrixes

    Returns
    -------
    [4,2,2] tensor of complex128
    """

    reduced_povm = tf.einsum('ijk,in->njk', sigma,
                             tf.cast(self.lam, dtype=tf.complex128))
    last_matrix_of_povm = sigma[0] - tf.einsum('ijk->jk',reduced_povm)
    return tf.concat([reduced_povm,last_matrix_of_povm[tf.newaxis]],
                     axis = 0)

  def measurement(self):
    """Initializes matrix of projective measurement in computational basis
    for pseudo POVM

    Returns
    -------
    [2,4] tensor of float64"""

    povm = self.get_pseudo_povm_in_comp_basis()
    T = tf.einsum('aij,bji->ab',povm,povm)
    T_inv = tf.linalg.inv(T)
    e = tf.einsum('lk,kij->lij',T_inv,povm)
    m0 = tf.constant([[1,0],[0,0]],dtype = tf.complex128)
    m1 = tf.constant([[0,0],[0,1]], dtype = tf.complex128)
    M = tf.concat([m0[tf.newaxis],m1[tf.newaxis]],axis=0)
    M_pr = tf.einsum('mij,dji->md',M,e)
    return tf.math.real(M_pr)

  def init_state(self):
    """Creates vector of state |0> in pseudo POVM

    Returns
    -------
    [4,.] tensor of float64
    """

    povm = self.get_pseudo_povm_in_comp_basis()
    ro_zero = tf.constant([[1,0],[0,0]],dtype=tf.complex128)
    return tf.math.real(tf.einsum('ij,kji->k',ro_zero,povm))

class QubitCircuit:
  """Class of one-qubit quantum circuit
  Contains quantum circuit as a list of quantum gates

  Attributes
  ----------
  list_of_gates
    list of [2,2] tensors of complex128

  Methods
  -------
  get_gates()
    returns list of quantum gates

  run_circuit(povm)
    runs the circuit in given POVM or pseudo POVM

  total_negativity(povm)
    computes total negativity of the circuit in given POVM or pseudo POVM
    including negativity of initial state and measurement matrix
  """

  def __init__(self, list_of_gates):
    self.list_of_gates = list_of_gates

  def get_gates(self):
    """Returns list of quantum gates"""

    return self.list_of_gates


  def run_circuit(self, povm):
    """Runs quantum circuit in given POVM or pseudo POVM

    Arguments
    ---------
    povm : PseudoPovm

    Returns [2, ] tensor of float64
    """

    gates = []
    gates_num = len(self.list_of_gates)
    for i in range(gates_num):
      gates.append(unitary_to_povm(self.list_of_gates[i],
                                   povm.get_pseudo_povm_in_comp_basis()))
    circuit = tf.eye(4,dtype=tf.float64)
    for i in range(gates_num):
      circuit = gates[i] @ circuit
    init_state = povm.init_state()
    measurement = povm.measurement()
    res = tf.einsum('ab,bc,c->a',measurement,circuit,init_state)
    return res

  def run_approx_circuit(self, povm):
    """Runs stochastic approximation of
    quantum circuit in given POVM or pseudo POVM

    Arguments
    ---------
    povm : PseudoPovm

    Returns [2, ] tensor of float64
    """

    gates = []
    gates_num = len(self.list_of_gates)
    for i in range(gates_num):
      gates.append(pseudostoch_approx(unitary_to_povm(self.list_of_gates[i],
                                   povm.get_pseudo_povm_in_comp_basis())))
    circuit = tf.eye(4,dtype=tf.float64)
    for i in range(gates_num):
      circuit = gates[i] @ circuit
    init_state = approx_vector(povm.init_state())
    measurement = pseudostoch_approx(povm.measurement())
    res = tf.einsum('ab,bc,c->a',measurement,circuit,init_state)
    return res


  def total_negativity(self, povm):
      """Computes total negativity of circuit including negativity of initial
      state and measurement

      Arguments
      ---------
      povm : PseudoPovm

      Returns scalar tensor of float64
      """

      i = tf.constant(0)

      neg = tf.constant(0,dtype = tf.float64)
      cond = lambda i, neg: i < len(self.list_of_gates)
      body = lambda i, neg: [i+1,
                             neg + negativity(unitary_to_povm(self.list_of_gates[i],
                                          povm.get_pseudo_povm_in_comp_basis()))]

      neg = tf.while_loop(cond, body, [i, neg])[1]

      neg += negativity(povm.measurement())
      neg += tf.math.log(tf.linalg.norm(povm.init_state(),ord = 1))
      return neg

def unitary_to_povm(unitary, povm):
  """Converts quantum single qubit gate to pseudostochastic matrix in
  given POVM or pseudo POVM

  Arguments
  ---------
  unitary : tensor [2,2] of complex128
    Note it can be not unitary gate
  povm : tensor [4,2,2] of complex128

  Returns tensor [4,4] of float64 corresponding to pseudostochastic matrix
  """

  T = tf.einsum('aij,bji->ab', povm, povm)
  T_inv = tf.linalg.inv(T)
  e = tf.einsum('lk,kij->lij', T_inv, povm)
  S = tf.einsum('abc,cd,zde,eb->az', povm, unitary, e,
                tf.linalg.adjoint(unitary))
  return tf.math.real(S)

def negativity(matrix):
  """Computes negativity of given matrix
  Negativity depends on negative elements in matrix
  It is logarythm of mean value of L1 norms of columns in matrix

  Arguments
  ---------
  matrix: any rank 2 tensor of real or complex numbers

  Returns scalar tensor of float64
  """

  negativity = tf.constant(0, dtype = tf.float64)
  columns = matrix.shape[1]

  i = tf.constant(0)
  cond = lambda i, negativity: i < columns
  body = lambda i, negativity: [i + 1,
                              negativity + tf.linalg.norm(matrix[:,i], ord = 1)]

  negativity = tf.while_loop(cond, body, [i, negativity])[1]

  return tf.math.log(negativity / columns)

def pseudostoch_approx(matrix):
  """
  Approximates pseudostochasic matrix with stochastic one

  Arguments
  ---------
  matrix : rank 2 tensor of tf.float

  Returns tensor of the same shape and dtype as matrix
  """
  A = matrix.numpy()
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      if A[i, j] < 0:
        A[i, j] = 0

  #normalize elements to create stochastic matrix
  s = np.zeros(A.shape[1]) #initializes list of sums in each column
  for j in range(A.shape[1]): #calculates sum of elements in each column
    s[j] = A[:,j].sum()
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      A[i,j] = (A[i,j] / s[j])

  return tf.constant(A)

def approx_vector(vector):
  """aproximetes pseudo probability vector with probability one

  Arguments
  ----------
  vector : rank 1 tensor of tf.float

  retunrs tensor of the same shape and dtype
  """
  A = vector.numpy()
  for i in range(A.shape[0]):
    if A[i] < 0:
      A[i] = 0
  s = A.sum()
  for i in range(A.shape[0]):
    A[i] = A[i] / s
  return tf.constant(A)
