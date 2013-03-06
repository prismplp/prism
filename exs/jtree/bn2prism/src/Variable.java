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

import java.util.*;

/** A variable in a Bayesian network. */
public class Variable
{
  /** Original variable name in the XML specification. */
  private String name = null;

  /** ID of the variable */
  private int id = -1;

  /** Original names of the outcomes in the XML specification. */
  private List<String> outcomeList = null;

  /** List of IDs of parent variables. */
  private List<Integer> parentList = null;

  /** CPT. */
  private double[] table = null;

  /** Original CPT. */
  private double[] origTable = null;

  /** The position of the variable in the total (topological) order. */
  private int order = -1;

  /** The index of an outcome that were observed as an evidence. */
  private int evidence = -1;

  /** Create a BN variable with no operation. */
  public Variable() {}

  public void setName(String n) { name = n; }
  public void setId(int i) { id = i; }
  public void setOrder(int o) { order = o; }
  public void setParentList(List<Integer> l) { parentList = l; }
  public void setEvidence(int e) { evidence = e; }
  public void setOutcomeList(List<String> l) { outcomeList = l; }

  public String getName() { return name; }
  public int getId() { return id; }
  public int getOrder() { return order; }
  public List<Integer> getParentList() { return parentList; }
  public int getEvidence() { return evidence; }
  public List<String> getOutcomeList() { return outcomeList; }

  public int getOutcomeSize()
  {
    return outcomeList.size();
  }

  public void addParent(int id)
  {
    if (parentList == null)
      parentList = new ArrayList<Integer>();
    parentList.add(id);
  }

  public void addOutcome(String o)
  {
    if (outcomeList == null)
      outcomeList = new ArrayList<String>();
    outcomeList.add(o);
  }

  public void setTable(double[] t)
  {
    table = t;
  }

  public double[] getTable()
  {
    return table;
  }

  /** Replaces the original CPT with a randomly generated one. */
  public void replaceRandomTable()
  {
    // Make a backup of the current CPTs:
    origTable = table;

    int numEntries = origTable.length;
    int size = getOutcomeSize();
    int numParInstances = numEntries / size;

    // Prepare an array for a random CPT:
    table = new double[numEntries];
    
    // Random number generator:
    Random rand = new Random();

    for (int row = 0; row < numParInstances; row++) {
      double sum = 0.0;
      for (int column = 0; column < size; column++) {
	int i = row * size + column;
	table[i] = rand.nextDouble();
	sum += table[i];
      }
      for (int column = 0; column < size; column++) {
	int i = row * size + column;
	table[i] /= sum;   // Normalize
      }
    }
  }

  public String toString()
  {
    StringBuilder sb = new StringBuilder();

    sb.append(name);
    sb.append("[");
    boolean isFirst = true;
    for (String out : outcomeList) {
      if (!isFirst) sb.append(",");
      sb.append(out);
      isFirst = false;
    }
    sb.append("]");

    sb.append(":<");
    isFirst = true;
    for (Integer i : parentList) {
      if (!isFirst) sb.append(",");
      sb.append(i);
      isFirst = false;
    }
    sb.append(">");

    sb.append(":<");
    isFirst = true;
    for (double d : table) {
      if (!isFirst) sb.append(",");
      sb.append(d);
      isFirst = false;
    }
    sb.append(">:");

    if (evidence >= 0)
      sb.append("<E" + evidence + ">");
    else
      sb.append("<>");

    return sb.toString();
  }

}
