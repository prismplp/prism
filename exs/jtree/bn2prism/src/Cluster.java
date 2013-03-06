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
 * A cluster used in the bucket-tree algorithm.  Note that clusters and
 * variables in the Bayesian network have a one-to-one mapping, so the ID
 * of a cluster is the same as the ID of the corresponding variable.
 */
public abstract class Cluster
{
  /** ID of the cluster. */
  int id;

  /**
   * Bit array that indicates the variables in the cluster.  That is,
   * <CODE>true</CODE> (resp. <CODE>false</CODE>) means the variable is
   * (resp. is not) included in the cluster.
   */
  VArray varAry;

  /**
   * The width of the cluster.  This is exactly the number of true bits in
   * <CODE>varAry</CODE>.
   */
  int width;

  /**
   * Bit array that indicates the input/output modes of the variables in the cluster.
   * <CODE>true</CODE> means the input (+) mode, and <CODE>false</CODE> means the
   * output (-) mode.
   */
  VArray modeAry; // Input-output mode -- true:+, false:-

  /**
   * Bit array that indicates the existences of the instanciation nodes in the
   * same bucket.
   */
  VArray instAry;

  /** Temporary variable used for breaking a loop of input/output dependency */
  int offset;

  public int getId()
  {
    return id;
  }

  public VArray getVarAry()
  {
    return varAry;
  }

  public int getWidth()
  {
    return width;
  }

  public void setModeAry(VArray m)
  {
    modeAry = m;
  }

  public void setInstAry(VArray ia)
  {
    instAry = ia;
  }

  public VArray getModeAry()
  {
    return modeAry;
  }

  public VArray getInstAry()
  {
    return instAry;
  }

  public void setOffset(int o)
  {
    offset = o;
  }

  public int getOffset()
  {
    return offset;
  }

  public abstract String toString();
  public abstract String toPrismString();
  public abstract String toPrismString2();

  protected int computeWidth(VArray va)
  {
    int w = 0;
    boolean[] a = va.getArray();
    for (int i = 0; i < a.length; i++) {
      if (a[i]) w++;
    }
    return w;
  }
}
