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
 * A cluster node used in the bucket-tree method.  In a probabilistic
 * context, a cluster node corresponds to a marginal distribution, and
 * the ID of a cluster node is the ID of the variable which has been
 * marginalized out lastly in the bucket-tree method.
 */
public class ClusterNode extends Cluster
{
  /**
   * @param i ID of the cluster node.
   * @param a bit array that indicates the variables in the cluster node.
   */
  public ClusterNode(int i, VArray a) throws B2PException
  {
    id = i;
    varAry = a;
    instAry = varAry.copy();
    instAry.setAll(false);
    width = computeWidth(a);
    offset = 0;
  }

  /** Returns an _internal_ string representation. */
  public String toString()
  {
    StringBuilder sb = new StringBuilder();

    sb.append("cl_" + id + "(");
    boolean isFirst = true;
    for (int i = 0; i < varAry.size(); i++) {
      if (varAry.get(i)) {
	if (!isFirst) sb.append(",");
	if (modeAry != null) sb.append((modeAry.get(i)) ? "+" : "-");
	sb.append(i);
	isFirst = false;
      }
    }
    sb.append(")");

    return sb.toString();
  }

  /**
   * Returns a mode declaration of the corresponding
   * "<CODE>node_x</CODE><I>N</I>" predicate.
   */
  public String toPrismModeString()
  {
    StringBuilder sb = new StringBuilder();

    sb.append("node_x" + id);
    boolean isFirst = true;
    for (int i = 0; i < varAry.size(); i++) {
      if (varAry.get(i)) {
	if (!isFirst)
	  sb.append(",");
	else
	  sb.append("(");
	if (modeAry != null) sb.append((modeAry.get(i)) ? "+" : "-");
	isFirst = false;
      }
    }
    if (!isFirst) sb.append(")");

    return sb.toString();

  }

  /**
   * Returns a mode declaration of the corresponding
   * "<CODE>pot_x</CODE><I>N</I>" predicate.
   */
  public String toPrismModeString2()
  {
    StringBuilder sb = new StringBuilder();

    sb.append("pot_x" + id + "(");
    boolean isFirst = true;
    for (int i = 0; i < varAry.size(); i++) {
      if (varAry.get(i) || i == id) {
	if (!isFirst) sb.append(",");
	if (modeAry != null) sb.append((modeAry.get(i)) ? "+" : "-");
	isFirst = false;
      }
    }
    sb.append(")");

    return sb.toString();
  }

  /**
   * Returns a string representation of the corresponding
   * "<CODE>node_x</CODE><I>N</I>" literal.
   */
  public String toPrismString()
  {
    StringBuilder sb = new StringBuilder();

    sb.append("node_x" + id);
    boolean isFirst = true;
    for (int i = 0; i < varAry.size(); i++) {
      if (varAry.get(i)) {
	if (!isFirst)
	  sb.append(",");
	else
	  sb.append("(");
	sb.append("_X" + i);
	isFirst = false;
      }
    }
    if (!isFirst) sb.append(")");

    return sb.toString();
  }

  /**
   * Returns a string representation of the corresponding
   * "<CODE>pot_x</CODE><I>N</I>" literal.
   */
  public String toPrismString2()
  {
    StringBuilder sb = new StringBuilder();

    sb.append("pot_x" + id + "(");
    boolean isFirst = true;
    for (int i = 0; i < varAry.size(); i++) {
      if (varAry.get(i) || i == id) {
	if (!isFirst) sb.append(",");
	sb.append("_X" + i);
	isFirst = false;
      }
    }
    sb.append(")");

    return sb.toString();
  }

}