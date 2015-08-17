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
 * An instanciation node, a special node which is newly introduced in
 * the BN2Prism translator.  An instanciation node corresponds to an
 * instanciation of a variable, and the ID of a instanciation node
 * is the ID of the instanciated variable.
 */
public class InstanciateNode extends Cluster
{
  /**
   * @param i ID of the instanciation node.
   */
  public InstanciateNode(int i)
  {
    id = i;
    varAry = null;
    width = 1;
  }

  /** Returns an _internal_ string representation. */
  public String toString()
  {
    return "inst(-" + id + ")";
  }

  /**
   * Returns a string representation of the corresponding
   * "<CODE>instanciate</CODE>" literal.
   */
  public String toPrismString()
  {
    return "instanciate(x" + id + ",_X" + id + ")";
  }

 /** The same as <CODE>toPrismString()</CODE>. */
   public String toPrismString2()
  {
    return toPrismString();
  }
}
