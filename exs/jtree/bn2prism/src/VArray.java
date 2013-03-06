//
// BN2Prism
// Copyright (c) 2007-2009, Yoshitaka Kameya
//
// All rights reserved.
//
// Redistribution and use in source and binary forms,  with or without
// modification,  are permitted provided that the following conditions
// are met:
// 
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions  in  binary  form  must  reproduce  the  above
//    copyright notice,  this list of conditions  and  the following
//    disclaimer  in  the  documentation  and/or   other   materials
//    provided with the distribution.
//  * Neither  the  name  of  Tokyo Institute of Technology  nor the
//    names of its contributors may be used to  endorse  or  promote
//    products derived from  this  software  without  specific prior
//    written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS"  AND  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO,  THE IMPLIED WARRANTIES OF MERCHANTABILITY  AND FITNESS
// FOR  A  PARTICULAR  PURPOSE  ARE  DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL,    SPECIAL,   EXEMPLARY,   OR   CONSEQUENTIAL   DAMAGES 
// (INCLUDING, BUT NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS;  OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT  LIABILITY,  OR  TORT  (INCLUDING  NEGLIGENCE  OR OTHERWISE)
// ARISING  IN  ANY  WAY  OUT  OF  THE  USE  OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

/**
 * A wrapper of a bit array related to the random variables
 * in a Bayesian network.
 */
public class VArray
{
  /** Target Bayesian network. */
  BayesNet bayesNet;

  /** Bit array. */
  boolean[] array;

  /**
   * Initializes all bits as b.
   * @param b  initial value for the bit array.
   */
  VArray(BayesNet bn, boolean b)
  {
    bayesNet = bn;
    int l = bayesNet.getNumberOfVariables();
    array = new boolean[l];

    for (int i = 0; i < l; i++) array[i] = b;
  }

  /**
   * Initializes all bits as false.
   * @param bn a Bayesian network
   */
  VArray(BayesNet bn)
  {
    this(bn, false);
  }

  /**
   * Sets true to the bits corresponding to the variables included
   * in the CPT at which the variable v appears.
   * @param bn a Bayesian network
   */
  VArray(BayesNet bn, Variable v)
  {
    this(bn);

    int l = array.length;

    for (int i = 0; i < l; i++)
      array[i] = false;

    array[v.getId()] = true;
    for (int j : v.getParentList())
      array[j] = true;
  }

  public boolean[] getArray()
  {
    return array;
  }

  /** Makes a deep copy of <CODE>a</CODE> to the bit array. */
  public void setArray(boolean[] a) throws B2PException
  {
    if (array.length != a.length)
      throw new B2PException("Different sizes");

    for (int i = 0; i < array.length; i++)
      array[i] = a[i];
  }

  /** Returns a copy of this object. */
  public VArray copy() throws B2PException
  {
    VArray va = new VArray(bayesNet);
    va.setArray(array);
    
    return va;
  }

  /** Returns the <CODE>i</CODE>-th element of the bit array. */
  public boolean get(int i)
  {
    return array[i];
  }

  /** Sets the <CODE>i</CODE>-th element of the bit array as <CODE>b</CODE>. */
  public void set(int i, boolean b)
  {
    array[i] = b;
  }

  /** Sets all elements of the bit array as <CODE>b</CODE>. */
  public void setAll(boolean b)
  {
    for (int i = 0; i < array.length; i++)
      array[i] = b;
  }

  /** Returns the size of the bit array. */
  public int size()
  {
    return array.length;
  }

  /** Returns a string representation of the bit array. */
  public String toString()
  {
    StringBuilder sb = new StringBuilder();
    final int SECTION_WIDTH = 10;

    sb.append("[");
    for (int i = 0; i < array.length; i++) {
      sb.append((array[i])? "1" : "0");
      if (i % SECTION_WIDTH == (SECTION_WIDTH - 1) && i < (array.length - 1))
	sb.append("|");
    }
    sb.append("]");

    return sb.toString();
  }

  /** Takes the bit-wise intersection of the bit array and <CODE>va</CODE>. */
  public void intersect(VArray va) throws B2PException
  {
    if (va == null || array.length != va.size())
      throw new B2PException("Illegal input");
    
    boolean[] a = va.getArray();
    for (int i = 0; i < array.length; i++)
      array[i] = array[i] & a[i];
  }

  /** Takes the bit-wise union of the bit array and <CODE>va</CODE>. */
  public void union(VArray va) throws B2PException
  {
    if (va == null || array.length != va.size())
      throw new B2PException("Illegal input");
    
    boolean[] a = va.getArray();
    for (int i = 0; i < array.length; i++)
      array[i] = array[i] | a[i];
  }

  /** Takes the bit-wise exclusive OR of the bit array and <CODE>va</CODE>. */
  public void xor(VArray va) throws B2PException
  {
    if (va == null || array.length != va.size())
      throw new B2PException("Illegal input");
    
    boolean[] a = va.getArray();
    for (int i = 0; i < array.length; i++)
      array[i] = array[i] ^ a[i];
  }

  /** Returns the bit-wise exclusive OR of <CODE>va1</CODE> and <CODE>va2</CODE>. */
  public static VArray intersect(VArray va1, VArray va2)
    throws B2PException
  {
    if (va1 == null || va2 == null || va1.size() != va2.size())
      throw new B2PException("Illegal input");

    VArray va = va1.copy();
    va.intersect(va2);

    return va;
  }

  /** Returns the bit-wise union of <CODE>va1</CODE> and <CODE>va2</CODE>. */
  public static VArray union(VArray va1, VArray va2)
    throws B2PException
  {
    if (va1 == null || va2 == null || va1.size() != va2.size())
      throw new B2PException("Illegal input");

    VArray va = va1.copy();
    va.union(va2);

    return va;
  }

  /** Returns the bit-wise exclusive OR of <CODE>va1</CODE> and <CODE>va2</CODE>. */
  public static VArray xor(VArray va1, VArray va2)
    throws B2PException
  {
    if (va1 == null || va2 == null || va1.size() != va2.size())
      throw new B2PException("Illegal input");

    VArray va = va1.copy();
    va.xor(va2);

    return va;
  }
}
