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
 * A CPT node used in the bucket-tree method.  The ID of a CPT node is
 * the ID of the child variable in the CPT.
 */
public class CPTNode extends Cluster
{
  /**
   * @param i ID of the cluster node.
   * @param a bit array that indicates the variables in the CPT.
   */
  public CPTNode(int i, VArray a) throws B2PException
  {
    id = i;
    varAry = a;
    instAry = varAry.copy();
    instAry.setAll(false);
    width = computeWidth(a);
    offset = 0;
  }

  /** Adds the input/output modes for all variables in the CPT. */
  public void buildModes()
  {
    for (int i = 0; i < varAry.size(); i++) {
      if (i != id && varAry.get(i)) // parent
	modeAry.set(i,true);
    }
  }

  /** Returns an _internal_ string representation. */
  public String toString()
  {
    StringBuilder sb = new StringBuilder();

    sb.append("p(");
    if (modeAry != null) sb.append("-");
    sb.append(id);
    boolean isFirst = true;
    for (int i = 0; i < varAry.size(); i++) {
      if (i != id && varAry.get(i)) { // parent
	if (!isFirst)
	  sb.append(",");
	else
	  sb.append("|");
	if (modeAry != null) sb.append((modeAry.get(i)) ? "+" : "-");
	sb.append(i);
	isFirst = false;
      }
    }
    sb.append(")");

    return sb.toString();
  }
  
  /**
   * Returns a string representation of the corresponding
   * "<CODE>cpt</CODE>" literal.
   */
  public String toPrismString()
  {
    StringBuilder sb = new StringBuilder();

    sb.append("cpt(");
    sb.append("x" + id + ",[");
    boolean isFirst = true;
    for (int i = 0; i < varAry.size(); i++) {
      if (i != id && varAry.get(i)) {
	if (!isFirst) sb.append(",");
	sb.append("x" + i + "=_X" + i);
	isFirst = false;
      }
    }
    sb.append("],_X" + id + ")");

    return sb.toString();
  }

  /** The same as <CODE>toPrismString()</CODE>. */
  public String toPrismString2()
  {
    return toPrismString();
  }
}