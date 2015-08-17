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

import javax.xml.parsers.*;
import org.w3c.dom.*;
import java.io.*;
import java.util.*;

/**
 * A Bayesian network.
 * Almost all translation procedures are done in this class. 
 */
public class BayesNet
{
  // Tag names in the input XML:
  private final static String TAG_BIF      = "BIF";
  private final static String TAG_NETWORK  = "NETWORK";
  private final static String TAG_NAME     = "NAME";
  private final static String TAG_VARIABLE = "VARIABLE";
  private final static String TAG_OUTCOME  = "OUTCOME";
  private final static String TAG_DEFINITION  = "DEFINITION";
  private final static String TAG_FOR         = "FOR";
  private final static String TAG_GIVEN       = "GIVEN";
  private final static String TAG_TABLE       = "TABLE";
  private final static String TAG_PROPERTY    = "PROPERTY";
  private final static String TAG_EVIDENCE    = "EVIDENCE";
  private final static String TAG_VARIABLENAME  = "VARIABLENAME";
  private final static String TAG_VALUE         = "VALUE";
  private final static String TAG_VALUES        = "VALUES";
  private final static String TAG_PROBABILITY   = "PROBABILITY";

  // Attribute names in the input XML:
  private final static String ATT_BIF_VERSION   = "VERSION";
  private final static String ATT_VARIABLE_TYPE = "TYPE";

  /** Enum for the flags in topological sorting. */
  private enum VStat { NEVER, JUST, ONCE }

  /** Number of levels for random evidences. */
  private final static int NUM_EVID_LEVEL = 4;

  /** Threshold by which we judge the non-tininess of a float-point number. */
  private final static double NON_TINY_PROB = 1.0e-3;

  /** Threshold by which we judge the tininess of a float-point number. */
  private final static double TINY_PROB = 1.0e-12;

  /** Max. number of variables allowed for a naive definition of a BN. */
  private final static int MAX_NVAR_NAIVE = 64;

  /** Basename of the input file. */
  private String basename;

  /** Version number of the translator. */
  private String version;

  /** Format of the XML input. */
  private Format format;

  /**
   * Flag indicating whether we have an external file (<CODE>BASENAME.num</CODE>)
   * which specifies the elimination order.
   */
  private boolean hasExtElimOrder;

  /** Flag indicating whether the CPTs given in the input file are transposed. */
  private boolean hasTrCPT;

  /** Flag for zero-compression of CPTs. */
  private boolean modeZeroComp;

  /** Flag for assigning random parameters. */
  private boolean modeRandom;

  /** Flag for generating random evidences. */
  private boolean modeEvid;

  /** Flag for normalizing CPT entries. */
  private boolean modeNormalize;

  /** Flag for the verbose mode. */
  private boolean modeVerbose;

  /** Flag indicating wheter the translator has some evidence file. */
  private boolean hasEvid;

  /** List of the variables appearing in the BN. */
  private List<Variable> varList;

  /** Hashtable for the variable names. */
  private Map<String,Integer> varMap;

  /** Directed adjacency table that specifies the network structure of the BN. */
  private int[][] adjTab;

  /** Undirected adjacency table that specifies the network structure of the BN. */
  private int[][] uAdjTab;

  /**
   * Elimination order. `<CODE>i = elimOrder[q]</CODE>' indicates that the q-th element
   * in the elimination order is variable <CODE>i</CODE>.
   */
  private int[] elimOrder;

  /**
   * List of buckets, where each buckets contains cluster nodes, CPT nodes and instanciation
   * nodes.
   */
  private List<List<Cluster>> bucketList;

  /**
   * List of cluster nodes that lead to be heads of the translated Prism clauses.
   */
  private Cluster[] headAry;

  /**
   * Total order of the variables in the BN.
   * This total order is obtained by topological sorting.
   */
  private int[] order;

  /** Inverse mapping of <CODE>order[]</CODE> (the total order). */
  private int[] invOrder;

  /**
   * Primary clusters for variables.  The primary cluster for a variable is the smallest
   * cluster to which the variable belongs.  Primary clusters are used in computing
   * the marginal distribution of a variable.
   */
  private int[] primCluster;

  /** The widths of primary clusters. */
  private int[] primClusterWidth;

  /** Output stream for the translated Prism program. */
  private PrintStream psm;

  /** Output stream for Prism evidences. */
  private PrintStream evidPsm;

  /** Output streams for random Prism evidences. */
  private PrintStream[] randEvidPsm;

  /**
   * Evidence levels used for generating random evidences.
   * The evidence level of variable <CODE>i</CODE> can be 1 ... <CODE>NUM_EVID_LEVEL</CODE>.
   * If the evidence level of variable <CODE>i</CODE> is <CODE>k</CODE>, the evidences
   * for variable i will be given in <CODE>k</CODE> files (i.e.
   * <CODE>BASENAME_revid{NUM_EVID_LEVEL - k + 1}.psm</CODE>,
   * ..., <CODE>BASENAME_revid{NUM_EVID_LEVEL}.psm</CODE>).
   */
  private int[] evidLevel;

  /**
   * @param setting Setting for the translator.
   * @param document DOM tree of the XML specification of a BN.
   * @param documentEvid DOM tree of the XML specification of evidences.
   */
  public BayesNet(BNSetting setting, Document document, Document documentEvid)
    throws B2PException
  {
    basename    = setting.getBaseName();
    version = setting.getVersion();
    format  = setting.getFormat();

    hasExtElimOrder = setting.getHasExtElimOrder();
    hasTrCPT        = setting.getHasTrCPT();
    modeZeroComp    = setting.getModeZeroComp();
    modeRandom      = setting.getModeRandom();
    modeEvid        = setting.getModeEvid();
    modeNormalize   = setting.getModeNormalize();
    modeVerbose     = setting.getModeVerbose();

    varList = new ArrayList<Variable>();
    varMap  = new HashMap<String,Integer>();
    adjTab  = null;
    uAdjTab = null;

    if (format == Format.XBIF)
      parseXBIFNetwork(document);
    else {
      parseXMLBIFNetwork(document);
      if (documentEvid != null) parseXMLBIFEvidence(documentEvid);
    }

    hasEvid = checkHasEvid();
    if (hasEvid) {
      try {
	evidPsm = new PrintStream(basename + "_evid.psm");
      }
      catch (Exception e) {
	throw new B2PException("Can't open output file(s): " + e.getMessage());
      }
    }

    try {
      psm = new PrintStream(basename + ".psm");

      if (modeEvid) {
	randEvidPsm = new PrintStream[NUM_EVID_LEVEL];
	for (int t = 0; t < NUM_EVID_LEVEL; t++)
	  randEvidPsm[t] = new PrintStream(basename + "_revid" + t + ".psm");
      }
    }
    catch (Exception e) {
      throw new B2PException("Can't open output file(s): " + e.getMessage());
    }
  }

  /** Checks if there are evidences specified in the input. */
  private boolean checkHasEvid()
  {
    boolean b = false;

    for (Variable v : varList) {
      if (v.getEvidence() >= 0) {
	b = true;
	break;
      }
    }

    return b;
  }

  /** Traverses and collects information on the DOM of XMLBIF _network_ file. */
  private void parseXMLBIFNetwork(Document document) throws B2PException
  {
    Node nn = getXMLBIFFirstNetworkNode(document);

    NodeList children = nn.getChildNodes();
    int l = children.getLength();

    int numVar = 0;
    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      if (cn.getNodeType() != Node.ELEMENT_NODE) continue;

      String s = cn.getNodeName();
      if (matchName(s,TAG_VARIABLE)) {
	addXMLBIFVariable(numVar,cn);
	numVar++;
      }
    }

    int numDef = 0;
    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      if (cn.getNodeType() != Node.ELEMENT_NODE) continue;

      String s = cn.getNodeName();
      if (matchName(s,TAG_DEFINITION)) {
	addXMLBIFDefinition(cn);
	numDef++;
      }
    }
    
    if (numVar != numDef) {
      String msg = "#variables (=" + numVar + ") and #CPTs (" + numDef + "differ.";
      throw new B2PException(msg);
    }

    postCheckNetwork();
  }

  /** Traverses and collects information on the DOM of XMLBIF _evidence_ file. */
  private void parseXMLBIFEvidence(Document document) throws B2PException
  {
    Node nn = getXMLBIFFirstNetworkNode(document);

    NodeList children = nn.getChildNodes();
    int l = children.getLength();

    int numEvid = 0;
    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      if (cn.getNodeType() != Node.ELEMENT_NODE) continue;

      String s = cn.getNodeName();
      if (matchName(s,TAG_EVIDENCE)) {
	addXMLBIFEvidence(numEvid,cn);
	numEvid++;
      }
    }
  }

  /** Traverses and collects information on the DOM of XBIF network file. */
  private void parseXBIFNetwork(Document document) throws B2PException
  {
    Node nn = getXBIFFirstNetworkNode(document);

    NodeList children = nn.getChildNodes();
    int l = children.getLength();

    int numVar = 0;
    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      if (cn.getNodeType() != Node.ELEMENT_NODE) continue;

      String s = cn.getNodeName();
      if (matchName(s,TAG_VARIABLE)) {
	addXBIFVariable(numVar,cn);
	numVar++;
      }
    }

    int numDef = 0;
    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      if (cn.getNodeType() != Node.ELEMENT_NODE) continue;

      String s = cn.getNodeName();
      if (matchName(s,TAG_PROBABILITY)) {
	addXBIFProbability(cn);
	numDef++;
      }
    }
    
    if (numVar != numDef) {
      String msg = "#variables (=" + numVar + ") and #CPTs (=" + numDef + ") differ.";
      throw new B2PException(msg);
    }

    postCheckNetwork();
  }

  /** Performs post checking and post processing of the network specification. */
  private void postCheckNetwork() throws B2PException
  {
    for (Variable v : varList) {
      if (v.getOutcomeList() == null)
	throw new B2PException("No outcome specification for variable " + v.getName());

      if (v.getTable() == null)
	throw new B2PException("No table specification for variable " + v.getName());
    }

    for (Variable v : varList) {
      List<Integer> parents = v.getParentList();
      if (parents == null)
	v.setParentList(new ArrayList<Integer>());
      else
	Collections.sort(parents);
    }

    for (Variable v : varList)
      checkTableSize(v);

    if (hasTrCPT)
      for (Variable v : varList)
	transposeCPT(v);  // Transpose the transposed CPT

    if (!modeRandom)
      for (Variable v : varList)
	checkTableEntries(v);

    if (modeRandom)
      for (Variable v : varList)
	v.replaceRandomTable();  // Replace the current CPT by a random CPT
  }

  /**
   * Checks the size of each CPT.  It is assumed that the parents and outcomes of
   * <CODE>v</CODE>, and the outcomes of <CODE>v</CODE>'s parents are non-null.
   */
  private void checkTableSize(Variable v) throws B2PException
  {
    int u = v.getOutcomeSize();
    
    for (int i : v.getParentList()) {
      u *= varList.get(i).getOutcomeSize();
    }

    if (u != v.getTable().length) {
      throw new B2PException("Unexpected number of table entries for variable: " + v);
    }
  }

  /** Transposes the transposed CPT. */
  private void transposeCPT(Variable v)
  {
    double[] pAry = v.getTable();
    int numEntries = pAry.length;
    double[] copy = new double[numEntries];
    int size = v.getOutcomeSize();
    int numParInstances = numEntries / size;  // # of instances of parents

    for (int i = 0; i < numEntries; i++) {
      int row = i / numParInstances;
      int column = i % numParInstances;
      int j = size * column + row;
      copy[j] = pAry[i];
    }

    v.setTable(copy);
  }

  /** Checks if the CPT entries are normalized. */
  private void checkTableEntries(Variable v) throws B2PException
  {
    double[] pAry = v.getTable();
    int numEntries = pAry.length;
    int size = v.getOutcomeSize();
    int numParInstances = numEntries / size;

    for (int row = 0; row < numParInstances; row++) {
      double sum = 0.0;
      for (int column = 0; column < size; column++) {
	int i = size * row + column;
	sum += pAry[i];
      }
      if (!modeRandom && !modeNormalize &&
	  (sum < 1.0 - NON_TINY_PROB || sum > 1.0 + NON_TINY_PROB)) {
	StringBuilder sb = new StringBuilder();
	if (size < 10) {
	  sb.append("[");
	  for (int column = 0; column < size; column++) {
	    if (column > 0) sb.append(", ");
	    int i = size * row + column;
	    sb.append(String.format("%12g",pAry[i]));
	  }
	  sb.append("]");
	}
	else {
	  sb.append("[...]");
	}
	String msg =
	  "Illegal CPT entries (for instance #" + row + " of parents) of variable " +
	  v.getName() + ": " + sb.toString() + "\n" + v;
	throw new B2PException(msg);
      }
      else if (sum >= 1.0 - TINY_PROB &&  sum <= 1.0 + TINY_PROB) {
	// If the CPT has a tolerable error, do nothing
      }
      else {
        if (modeVerbose && !modeRandom && !modeNormalize) {
          StringBuilder sb = new StringBuilder();
          if (size < 5) {
            sb.append("[");
            for (int column = 0; column < size; column++) {
              if (column > 0) sb.append(", ");
              int i = size * row + column;
              sb.append(String.format("%12g",pAry[i]));
            }
            sb.append("]");
          }
          else {
            sb.append("[..(too large)..]");
          }
          String msg =
            "Warning: normalized the CPT entries (for instance #" + row +
            " of parents) of variable " + v.getName() + ": " + sb.toString();
          System.err.println(msg);
        }
	for (int column = 0; column < size; column++) {
	  int i = size * row + column;
	  pAry[i] /= sum;
	}
      }
    }
  }

  /** Closes the output streams. */
  public void closeStreams()
  {
    psm.close();

    if (hasEvid) evidPsm.close();

    if (modeEvid)
      for (int t = 0; t < NUM_EVID_LEVEL; t++)
        randEvidPsm[t].close();
  }

  /**
   * Gets the DOM tree of the XMLBIF specification of a BN.
   * Note that we only get the first Bayesian network here.
   */
  private Node getXMLBIFFirstNetworkNode(Document document) throws B2PException
  {
    return getFirstNetworkNode(document);
  }

  /**
   * Gets the DOM tree of the XBIF specification of a BN.
   * Note that we only get the first Bayesian network here.
   */
  private Node getXBIFFirstNetworkNode(Document document) throws B2PException
  {
    return getFirstNetworkNode(document);
  }

  /** Gets the first DOM tree of the BN specification. */
  private Node getFirstNetworkNode(Document document) throws B2PException
  {
    Element root = document.getDocumentElement();

    NodeList children = root.getChildNodes();
    int l = children.getLength();

    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      if (cn.getNodeType() == Node.ELEMENT_NODE)
	return cn;
    }

    throw new B2PException("No <NETWORK> tag found");
  }

  /** Finds the definitions of random variables and adds them to the DOM tree. */
  private void addXMLBIFVariable(int id, Node n) throws B2PException
  {
    NodeList children = n.getChildNodes();
    String name = null;
    List<String> ol = new ArrayList<String>();
    int l = children.getLength();

    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      String s = cn.getNodeName();
      if (matchName(s,TAG_NAME)) {
	Node gcn = cn.getFirstChild();
	if (gcn.getNodeType() == Node.TEXT_NODE) {
	  name = getCleanTextContent(gcn);
	}
	else throw new B2PException("Unexpected element in <" + TAG_NAME + "> tag");
      }
      else if (matchName(s,TAG_OUTCOME)) {
	Node gcn = cn.getFirstChild();
	if (gcn.getNodeType() == Node.TEXT_NODE) {
	  String outcome = getCleanTextContent(gcn);
	  ol.add(outcome);
	}
	else throw new B2PException("Unexpected element in <" + TAG_OUTCOME + "> tag");
      }
    }

    if (name != null || ol.size() > 0) {
      Variable v = new Variable();
      v.setName(name);
      v.setId(id);
      v.setOutcomeList(ol);
      varList.add(v);
      varMap.put(name,id);
    }
    else throw new B2PException("Invalid variable specification.");
  }

  /** Finds CPTs and adds them to the DOM tree. */
  private void addXMLBIFDefinition(Node n) throws B2PException
  {
    NodeList children = n.getChildNodes();
    int l = children.getLength();

    String varFor = null;
    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      String s = cn.getNodeName();
      if (matchName(s,TAG_FOR)) {
	Node gcn = cn.getFirstChild();
	if (gcn.getNodeType() == Node.TEXT_NODE) {
	  varFor = getCleanTextContent(gcn);
	}
	else throw new B2PException("Unexpected element in <" + TAG_FOR + "> tag");
      }
    }
    if (varFor == null)
      throw new B2PException("No <" + TAG_FOR + "> tag");

    if (varMap.get(varFor) == null)
      throw new B2PException("Unknown variable for <" + TAG_FOR + "> tag: " + varFor);

    Variable var = varList.get(varMap.get(varFor));

    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      String s = cn.getNodeName();
      if (matchName(s,TAG_GIVEN)) {
	Node gcn = cn.getFirstChild();
	if (gcn.getNodeType() == Node.TEXT_NODE) {
	  String varGiven = getCleanTextContent(gcn);
	  var.addParent(varMap.get(varGiven));
	}
	else throw new B2PException("Unexpected element in <" + TAG_GIVEN + "> tag");
      }
    }

    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      String s = cn.getNodeName();
      if (matchName(s,TAG_TABLE)) {
	Node gcn = cn.getFirstChild();
	if (gcn.getNodeType() == Node.TEXT_NODE) {
	  String tab = getCleanTextContent(gcn);
	  String[] pStrAry = tab.split(" ");
	  double[] pAry = new double[pStrAry.length];
	  int k = 0;
	  for (String pStr : pStrAry) {
	    pAry[k] = Double.parseDouble(pStr);
	    k++;
	  }
	  var.setTable(pAry);
	}
	else throw new B2PException("Unexpected element in <" + TAG_TABLE + "> tag");
      }
    }
  }

  /** Finds evidences and adds them to the DOM tree. */
  private void addXMLBIFEvidence(int id, Node n) throws B2PException
  {
    NodeList children = n.getChildNodes();
    int l = children.getLength();
    String name = null;
    String value = null;

    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      String s = cn.getNodeName();
      if (matchName(s,TAG_VARIABLENAME)) {
	Node gcn = cn.getFirstChild();
	if (gcn.getNodeType() == Node.TEXT_NODE) {
	  name = getCleanTextContent(gcn);
	}
	else throw new B2PException("Unexpected element in <" + TAG_VARIABLENAME + "> tag");
      }
      else if (matchName(s,TAG_VALUE)) {
	Node gcn = cn.getFirstChild();
	if (gcn.getNodeType() == Node.TEXT_NODE) {
	  value = getCleanTextContent(gcn);
	}
	else throw new B2PException("Unexpected element in <" + TAG_VARIABLENAME + "> tag");
      }
    }

    if (name != null || value != null) {
      boolean found = false;
      for (Variable v : varList) {
	if (v.getName().equals(name)) {
	  found = true;
	  for (int i = 0; i < v.getOutcomeSize(); i++) {
	    if (v.getOutcomeList().get(i).equals(value)) {
	      v.setEvidence(i);
	      break;
	    }
	  }
	  if (v.getEvidence() < 0) found = false;
	  break;
	}
      }

      if (!found) {
	String msg = "Evidence (" + name + ", " + value + ") is not found";
	throw new B2PException(msg);
      }
    }
  }

  /** Finds the definitions of random variables and adds them to the DOM tree. */
  private void addXBIFVariable(int id, Node n) throws B2PException
  {
    NodeList children = n.getChildNodes();
    String name = null;
    int l = children.getLength();
    int numOutcomes = -1;

    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      String s = cn.getNodeName();
      if (matchName(s,TAG_NAME)) {
	Node gcn = cn.getFirstChild();
	if (gcn.getNodeType() == Node.TEXT_NODE) {
	  name = getCleanTextContent(gcn);
	}
	else throw new B2PException("Unexpected element in <" + TAG_NAME + "> tag");
      }
      else if (matchName(s,TAG_VALUES)) {
	Node gcn = cn.getFirstChild();
	if (gcn.getNodeType() == Node.TEXT_NODE) {
	  numOutcomes = Integer.parseInt(getCleanTextContent(gcn));
	}
	else throw new B2PException("Unexpected element in <" + TAG_OUTCOME + "> tag");
      }
    }

    if (name != null || numOutcomes > 0) {
      Variable v = new Variable();
      v.setName(name);
      v.setId(id);
      List<String> ol = new ArrayList<String>();
      for (int i = 0; i < numOutcomes; i++) {
	ol.add(String.valueOf(i));
      }
      v.setOutcomeList(ol);
      varList.add(v);
      varMap.put(name,id);
    }
    else throw new B2PException("Invalid variable specification.");
  }

  /** Finds CPTs and adds them to the DOM tree. */
  private void addXBIFProbability(Node n) throws B2PException
  {
    NodeList children = n.getChildNodes();
    int l = children.getLength();

    String varFor = null;
    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      String s = cn.getNodeName();
      if (matchName(s,TAG_FOR)) {
	Node gcn = cn.getFirstChild();
	if (gcn.getNodeType() == Node.TEXT_NODE) {
	  varFor = getCleanTextContent(gcn);
	}
	else throw new B2PException("Unexpected element in <" + TAG_FOR + "> tag");
      }
    }
    if (varFor == null)
      throw new B2PException("No <" + TAG_FOR + "> tag");

    if (varMap.get(varFor) == null)
      throw new B2PException("Unknown variable for <" + TAG_FOR + "> tag: " + varFor);

    Variable var = varList.get(varMap.get(varFor));

    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      String s = cn.getNodeName();
      if (matchName(s,TAG_GIVEN)) {
	Node gcn = cn.getFirstChild();
	if (gcn.getNodeType() == Node.TEXT_NODE) {
	  String varGiven = getCleanTextContent(gcn);
	  var.addParent(varMap.get(varGiven));
	}
	else throw new B2PException("Unexpected element in <" + TAG_GIVEN + "> tag");
      }
    }

    for (int i = 0; i < l; i++) {
      Node cn = children.item(i);
      String s = cn.getNodeName();
      if (matchName(s,TAG_TABLE)) {
	Node gcn = cn.getFirstChild();
	if (gcn.getNodeType() == Node.TEXT_NODE) {
	  String tab = getCleanTextContent(gcn);
	  String[] pStrAry = tab.split(" ");
	  double[] pAry = new double[pStrAry.length];
	  int k = 0;
	  for (String pStr : pStrAry) {
	    pAry[k] = Double.parseDouble(pStr);
	    k++;
	  }
	  var.setTable(pAry);
	}
	else throw new B2PException("Unexpected element in <" + TAG_TABLE + "> tag");
      }
    }
  }

  /** Matches two strings ignoring the case. */
  private boolean matchName(String s, String name) 
  {
    return s.toUpperCase().equals(name.toUpperCase());
  }

  /**
   * Cleans up the text contents by replacing all space symbols by white spaces
   * and perform triming.
   */
  private String getCleanTextContent(Node n)
  {
    return n.getTextContent().replaceAll("\\s+"," ").trim();
  }

  /** Returns the number of variables in the BN. */
  public int getNumberOfVariables()
  {
    return varList.size();
  }

  /** Returns the i-th variable in the BN. */
  public Variable getVariable(int i)
  {
    return varList.get(i);
  }

  /** Builds the directed adjacency table and return it. */
  public int[][] getAdjacencyTable()
  {
    if (adjTab != null) return adjTab;

    int l = getNumberOfVariables();
    adjTab = new int[l][l];
    
    for (int i = 0; i < l; i++) {
      for (int j = 0; j < l; j++) {
	adjTab[i][j] = 0;
      }
    }

    for (int i = 0; i < l; i++) {
      Variable v = varList.get(i);
      for (int j : v.getParentList()) {
	adjTab[i][j] = -1;
	adjTab[j][i] = 1;
      }
    }

    return adjTab;
  }

  /** Prints the directed adjacency table. */
  public void printAdjacencyTable()
  {
    int[][] t = getAdjacencyTable();
    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();

    sb.append("%%%%\n");
    sb.append("%%%% Adjacency table:\n");
    sb.append("%%%%\n");

    for (int i = 0; i < l; i++) {
      sb.append("%% ");
      sb.append(String.format("%4d:",i));
      sb.append("[");
      for (int j = 0; j < l; j++) {
	//if (j > 0) sb.append(" ");
	if (t[i][j] > 0)
	  sb.append("+");
	else if (t[i][j] < 0)
	  sb.append("-");
	else if (i == j)
	  sb.append("\\");
	else
	  sb.append(0);
      }
      sb.append("]\n");
    }
    sb.append("\n");
    psm.print(sb);
  }

  /** Prints the undirected adjacency table. */
  public void printUndirectedAdjacencyTable()
  {
    int[][] t = getUndirectedAdjacencyTable();
    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < l; i++) {
      sb.append(String.format("%4d:",i));
      sb.append("[");
      for (int j = 0; j < l; j++) {
	//if (j > 0) sb.append(" ");
	if (t[i][j] > 0)
	  sb.append(1);
	else
	  sb.append(0);
      }
      sb.append("]\n");
    }
    sb.append("\n");

    psm.print(sb);
  }

  /** Converts a directed adjacency table to the undirected one. */
  public int[][] getUndirectedAdjacencyTable()
  {
    if (uAdjTab != null) return uAdjTab;

    int l = getNumberOfVariables();
    uAdjTab = new int[l][l];

    // Copy first:
    for (int i = 0; i < l; i++) {
      for (int j = 0; j < l; j++) {
	uAdjTab[i][j] = adjTab[i][j];
      }
    }

    int[] parent = new int[l];
    for (int i = 0; i < l; i++) {
      int k = 0;
      for (int j = 0; j < l; j++) {
	if (adjTab[i][j] < 0) {
	  parent[k] = j;
	  k++;
	}
      }
      int numParents = k;
      for (int k1 = 0; k1 < numParents; k1++) {
	int j1 = parent[k1];
	for (int k2 = 0; k2 < numParents; k2++) {
	  int j2 = parent[k2];
	  if (j1 != j2)
	    uAdjTab[j1][j2] = 1;
	}
      }
    }

    for (int i = 0; i < l; i++) {
      for (int j = 0; j < l; j++) {
	if (uAdjTab[i][j] != 0) uAdjTab[i][j] = 1;
      }
    }

    return uAdjTab;
  }

  /**
   * Gets the elimination order.  If it is not given in the file named
   * BASENAME.num, we compute MDO (minimum deficiency order), which is described in:
   * <P>
   * B. D'Ambrosio (1999).  Inference in Bayesian networks.  <I>AI Magazine</I>,
   * 20 (2), pp.21-36.
   * </P>
   */
  public void getElimOrder() throws B2PException
  {
    if (hasExtElimOrder)
      getExternalElimOrder();
    else
      buildElimOrder();
  }

  /** Retrieves the elimination order from the file named BASENAME.num. */
  private void getExternalElimOrder() throws B2PException
  {
    String ordfile = basename + ".num";
    String msg = null;
    List<String> varNameList = new ArrayList<String>();

    try {
      FileInputStream fis = new FileInputStream(ordfile);
      InputStreamReader ir = new InputStreamReader(fis);
      BufferedReader br = new BufferedReader(ir);

      while ((msg = br.readLine()) != null) {
	String[] varNames = msg.split("\\s+");
	for (String varName : varNames) {
	  varNameList.add(varName);
	}
      }

    }
    catch (Exception e) {
      throw new B2PException("Can't read the elimination order from " + ordfile + ": " +
			     e.getMessage());
    }
    
    int l = getNumberOfVariables();

    if (l != varNameList.size()) {
      String s =
	"Illegal specification of elimination order (#Variables = " + l +
	" / |elimination order| = " + varNameList.size() + ")";
      throw new B2PException(s);
    }

    elimOrder = new int[l];

    for (int q = 0; q < l; q++) {
      elimOrder[q] = -1;
    }
    
    for (int q = 0; q < l; q++) {
      elimOrder[q] = varMap.get(varNameList.get(q));
    }

    for (int q = 0; q < l; q++) {
      if (elimOrder[q] < 0 || elimOrder[q] >= l)
	throw new B2PException("Illegal external elimination order (obtained from " + ordfile + ")");
    }
  }

  /** Builds the minimum deficiency order (MDO) using adjacency tables and return it. */
  private void buildElimOrder()
  {
    int l = getNumberOfVariables();

    elimOrder = new int[l];

    boolean[] processed = new boolean[l];
    int[] score = new int[l];
    int[] neighbor = new int[l];
    int numNeighbors;
    int[][] uAdjTabCopy = new int[l][l];

    for (int i = 0; i < l; i++) {
      processed[i] = false;
    }

    // Copy the undirected adjacency table:
    for (int i = 0; i < l; i++) {
      for (int j = 0; j < l; j++) {
	uAdjTabCopy[i][j] = uAdjTab[i][j];
      }
    }

    // [NOTE] q is the index for elimOrder[]:
    for (int q = 0; q < l; q++) {

      for (int i = 0; i < l; i++) {
	if (processed[i]) continue;
	
	int k = 0;
	for (int j = 0; j < l; j++) {
	  if (!processed[j] && uAdjTabCopy[i][j] != 0) {
	    neighbor[k] = j;
	    k++;
	  }
	}
	numNeighbors = k;
	
	int e = 0;
	for (int k1 = 0; k1 < numNeighbors; k1++) {
	  int j1 = neighbor[k1];
	  for (int k2 = (k1 + 1); k2 < numNeighbors; k2++) {
	    int j2 = neighbor[k2];
	    if (uAdjTabCopy[j1][j2] != 0) e++;
	  }
	}
	int numExistEdges = e;
	
	int numFullEdges = numNeighbors * (numNeighbors - 1) / 2;
	score[i] = numFullEdges - numExistEdges;
      }
      
      int minScore = -1;
      int minI = -1;
      for (int i = 0; i < l; i++) {
	if (processed[i]) continue;
	if (minScore < 0 || score[i] < minScore) {
	  minI = i;
	  minScore = score[i];
	}
      }

      elimOrder[q] = minI;
      processed[minI] = true;

      if (minScore > 0) {
	int k = 0;
	for (int j = 0; j < l; j++) {
	  if (!processed[j] && uAdjTabCopy[minI][j] != 0) {
	    neighbor[k] = j;
	    k++;
	  }
	}
	numNeighbors = k;

	for (int k1 = 0; k1 < numNeighbors; k1++) {
	  int j1 = neighbor[k1];
	  for (int k2 = (k1 + 1); k2 < numNeighbors; k2++) {
	    int j2 = neighbor[k2];
	    if (uAdjTabCopy[j1][j2] == 0) {
	      uAdjTabCopy[j1][j2] = 1;
	      uAdjTabCopy[j2][j1] = 1;
	    }
	  }
	}
      }
      
    }

  }

  /** Prints the elimination order. */
  public void printElimOrder() throws B2PException
  {
    if (elimOrder == null) getElimOrder();
    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();

    sb.append("Elimination order = [");
    for (int q = 0; q < l; q++) {
      if (q > 0) sb.append(",");
      sb.append(elimOrder[q]);
    }
    sb.append("]\n\n");

    psm.print(sb);
  }

  /** Builds the initial buckets. */
  public void buildInitialBuckets() throws B2PException
  {
    int l = getNumberOfVariables();

    // Create a list of empty buckets:
    bucketList = new ArrayList<List<Cluster>>();
    for (int i = 0; i < l; i++) {
      bucketList.add(new ArrayList<Cluster>());
    }

    boolean[] processed = new boolean[l];
    for (int i = 0; i < l; i++) {
      processed[i] = false;
    }

    for (int q = 0; q < l; q++) {
      int i = elimOrder[q];
      List<Cluster> bucket = bucketList.get(q);

      if (!processed[i]) {
	VArray va = new VArray(this, varList.get(i));
	Cluster cl = new CPTNode(i,va);
	bucket.add(cl);
	processed[i] = true;
      }

      for (int j = 0; j < l; j++) {
	if (adjTab[j][i] < 0 && !processed[j]) {
	  VArray va = new VArray(this, varList.get(j));
	  Cluster cl = new CPTNode(j,va);
	  bucket.add(cl);
	  processed[j] = true;
	}
      }
    }

    //printBuckets();
  }

  /**
   * Grows the buckets one by one.  The procedure is based on the bucket-tree
   * method described in:
   * <P>
   * K. Kask, R. Dechter, J. Larrosa and A. Dechter (2005).
   * Unifying tree decompositions for reasoning in graphical models.
   * <I>Artificial Intelligence</I>, 166 (1-2), pp.165-193.
   * </P>
   */
  public void growBuckets() throws B2PException
  {
    int l = getNumberOfVariables();

    // Create a list of heads:
    headAry = new Cluster[l];

    for (int q = 0; q < l; q++) {
      int i = elimOrder[q];
      
      VArray head_va = new VArray(this);

      List<Cluster> bucket = bucketList.get(q);
      for (Cluster cl : bucket)
	head_va.union(cl.getVarAry());

      head_va.set(i,false);

      Cluster head = new ClusterNode(i,head_va);
      headAry[q] = head;

      for (int r = (q+1); r < l; r++) {
	int j = elimOrder[r];
	if (head_va.get(j)) {
	  bucketList.get(r).add(head);
	  break;
	}
      }
    }

    //printBuckets();
  }

  /** Determines the input/output modes of cluster nodes and CPT nodes. */
  public void buildModes() throws B2PException
  {
    int l = getNumberOfVariables();

    for (int q = 0; q < l; q++) {
      List<Cluster> bucket = bucketList.get(q);
      for (Cluster cl : bucket)
	cl.setModeAry(new VArray(this));
    }

    for (int q = 0; q < l; q++) {
      int i = elimOrder[q];

      List<Cluster> bucket = bucketList.get(q);
      for (Cluster cl : bucket) {
	if (cl instanceof CPTNode)
	  ((CPTNode)cl).buildModes();
      }

      VArray modes = new VArray(this);

      for (Cluster cl : bucket)
	modes.union(cl.getVarAry());

      for (Cluster cl : bucket) {
	VArray va = cl.getVarAry();
	VArray m = cl.getModeAry();
	for (int j = 0; j < modes.size(); j++) {
	  if (va.get(j) && !m.get(j))
	    modes.set(j,false);
	}
      }

      headAry[q].setModeAry(modes);
    }
  }

  /**
   * Orders the elements in each bucket according to the input/output modes.
   * To do this, we basically perform a topological sorting.  Besides, if a
   * dependency loop is found while sorting, we add an instanciation node
   * to break such a loop.
   */
  public void sortBuckets() throws B2PException
  {
    int l = getNumberOfVariables();

    for (int q = 0; q < l; q++) {
      List<Cluster> bucket = bucketList.get(q);
      int bSize = bucket.size();
      
      /* System.out.println("{Bucket " + elimOrder[q] + "}"); */

      int[][] bAdjTab = getModeDepTable(bSize, bucket);
      VStat[] bVisited = new VStat[bSize];
      boolean hasLoop = true;
      List<Integer> bOrder = null;

      while (hasLoop) {
	for (int i = 0; i < bSize; i++)
	  bVisited[i] = VStat.NEVER;
	
	VisitResult r = new VisitResult();
	bOrder = new ArrayList<Integer>();
	hasLoop = false;

	for (int i = 0; i < bSize; i++) {
	  if (bVisited[i] == VStat.NEVER) { 
	    r = visitBucket(bSize, i, r, bAdjTab, bVisited, bOrder);
	    if (r.failed()) {
	      Cluster loopy = bucket.get(r.getIndex());
	      VArray modes = loopy.getModeAry();
	      for (int k = loopy.getOffset(); k < l; k++) {
		if (modes.get(k)) {
		  bucket.add(new InstanciateNode(k));   // a variable in a loop
		  for (Cluster cl : bucket) {
		    if (!(cl instanceof InstanciateNode))
		      cl.getInstAry().set(k,true);
		  }
		  loopy.setOffset(k + 1);  // set the next start point 
		  break;
		}
	      }
	      bAdjTab = getModeDepTable(bSize, bucket); // update the table
	      hasLoop = true;
	      break;
	    }
	  }
	}
      }

      List<Cluster> sortedBucket = new ArrayList<Cluster>();
      VArray modes = headAry[q].getModeAry();

      for (Cluster cl : bucket) {
	if (cl instanceof InstanciateNode) {
	  sortedBucket.add(cl);
	  modes.set(cl.getId(), false);
	}
      }
      for (int i : bOrder)
	sortedBucket.add(bucket.get(i));

      bucketList.set(q, sortedBucket);  // just replaced with the sorted one
    }
  }

  /**
   * Visit an element in a bucket recursively.  This is a depth-first recursive
   * routine for topological sorting.
   */
  private VisitResult visitBucket(int bSize, int i, VisitResult r,
				  int[][] adj, VStat[] visited, List<Integer> order)
  {
    visited[i] = VStat.JUST;

    for (int j = 1; j < bSize; j++) {
      if (adj[j][i] <= 0) continue;
      if (visited[j] == VStat.NEVER) {
	r = visitBucket(bSize, j, r, adj, visited, order);
	if (r.failed()) return r;
      }
      else if (visited[j] == VStat.JUST) {
	return new VisitResult(false, j);
      }
    }

    order.add(i);
    visited[i] = VStat.ONCE;

    return new VisitResult(true, i);
  }

  /**
   * Gets the table on input/output dependencies.  Note that `bSize' is the size of
   * the original bucket and `bucket' may contain some instanciation nodes after
   * the (bSize)-th element.
   */
  private int[][] getModeDepTable(int bSize, List<Cluster> bucket)
    throws B2PException
  {
    int l = getNumberOfVariables();

    int[][] bAdjTab = new int[bSize][bSize];
    for (int i = 0; i < bSize; i++)
      for (int j = i; j < bSize; j++)
	bAdjTab[i][j] = 0;

    for (int i = 0; i < bSize; i++) {
      Cluster cl1 = bucket.get(i);
      VArray va1 = cl1.getVarAry();
      VArray m1 = cl1.getModeAry();
      VArray inst1 = cl1.getInstAry();

      for (int j = (i + 1); j < bSize; j++) {
	Cluster cl2 = bucket.get(j);
	VArray va2 = cl2.getVarAry();
	VArray m2 = cl2.getModeAry();
	VArray inst2 = cl2.getInstAry();

	VArray m = VArray.xor(m1,m2);

	int c1 = 0;  // # of input variables in m1 that form a loop
	int c2 = 0;  // # of input variables in m2 that form a loop
	for (int k = 0; k < l; k++) {
	  if (!(va1.get(k) && va2.get(k))) continue;
	  if (!m.get(k)) continue;
	  if (inst1.get(k) || inst2.get(k)) continue;
	  if (m1.get(k))
	    c1++;
	  else
	    c2++;
	}

	// No overlaps
	if (c1 == 0 && c2 == 0) continue;

	// If there are loops among two clusters, add instianciation nodes
	// to break such loops:
	if (c1 > 0 && c2 > 0) {
	  for (int k = 0; k < l; k++) {
	    if (!(va1.get(k) && va2.get(k))) continue;
	    if (!m.get(k)) continue;
	    if (inst1.get(k) || inst2.get(k)) continue;
	    if ((c1 < c2) && m1.get(k)) {
	      bucket.add(new InstanciateNode(k));
	      inst1.set(k,true);
	      inst2.set(k,true);
	      c1--;
	    }
	    if ((c1 >= c2) && m2.get(k)) {
	      bucket.add(new InstanciateNode(k));
	      inst1.set(k,true);
	      inst2.set(k,true);
	      c2--;
	    }
	  }
	}
	
	if (c1 >= c2) { // Flow: Variable j => i
	  bAdjTab[i][j] = -1;
	  bAdjTab[j][i] =  1;
	}
	else {          // Flow: Variable i => j
	  bAdjTab[i][j] =  1;
	  bAdjTab[j][i] = -1;
	}
      }

    }

    return bAdjTab;
  }

  /** Immutable class containing a result of a visit in sorting a bucket. */
  private class VisitResult {
    boolean succ;  // succeeded or not
    int index;     // index of the visited cluster

    VisitResult() {
      this(true,-1);
    }

    VisitResult(boolean s, int i) {
      succ = s;
      index = i;
    }

    public boolean failed() {
      return !succ;
    }

    public int getIndex() {
      return index;
    }
  }

  /** Prints the initial buckets. */
  public void printInitialBuckets()
  {
    StringBuilder sb = new StringBuilder();

    sb.append("%%%%\n");
    sb.append("%%%% Bucket Tree (initial)\n");
    sb.append("%%%%\n");
    psm.print(sb);

    printBuckets();
  }

  /** Prints the buckets finally obtained by the bucket-tree method. */
  public void printFinalBuckets()
  {
    StringBuilder sb = new StringBuilder();

    sb.append("%%%%\n");
    sb.append("%%%% Bucket Tree (final)\n");
    sb.append("%%%%\n");
    psm.print(sb);

    printBuckets();
  }

  /** Prints the buckets added the input/output modes. */
  public void printModedBuckets()
  {
    StringBuilder sb = new StringBuilder();

    sb.append("%%%%\n");
    sb.append("%%%% Bucket Tree (modes)\n");
    sb.append("%%%%\n");
    psm.print(sb);

    printBuckets();
  }

  /** Prints buckets (the base method). */
  public void printBuckets()
  {
    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();

    for (int q = 0; q < l; q++) {
      int i = elimOrder[q];
      sb.append("%% Bucket [" + i + "] (" + q + "): ");

      List<Cluster> bucket = bucketList.get(q);
      boolean isFirst1 = true;
      sb.append("{ ");
      for (Cluster c : bucket) {
	if (!isFirst1) sb.append("; ");
	sb.append(c);
	isFirst1 = false;
      }
      sb.append(" }");

      if (headAry != null) {
	sb.append(" => ");
	Cluster head = headAry[q];
	if (head != null)
	  sb.append(head);
	else
	  sb.append("undef");
      }

      sb.append("\n");
    }
    sb.append("\n");

    psm.print(sb);
  }

  /** Writes the version number to the outoput. */
  public void printPrismVersion()
  {
    StringBuilder sb = new StringBuilder();
    
    sb.append("%%%%\n");
    sb.append("%%%% Junction-tree PRISM program for `" + basename + "' generated by BN2Prism " + version + "\n");
    sb.append("%%%%\n\n");

    psm.print(sb);
  }

  /** Writes the mode declarations to the outoput. */
  public void printPrismModes()
  {
    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();

    sb.append("%%%%\n");
    sb.append("%%%% Mode declarations\n");
    sb.append("%%%%\n\n");

    for (int q = (l - 1); q >= 0; q--) {
      int i = elimOrder[q];
      ClusterNode head = (ClusterNode)headAry[q];
      if (head.getWidth() > 0)
	sb.append(":- mode " + head.toPrismModeString() + ".\n");
    }
    sb.append("\n");

    for (int q = (l - 1); q >= 0; q--) {
      int i = elimOrder[q];
      ClusterNode head = (ClusterNode)headAry[q];
      sb.append(":- mode " + head.toPrismModeString2() + ".\n");
    }
    sb.append("\n");

    psm.print(sb);
  }

  /** Writes the declaration part to the outoput. */
  public void printPrismDecls()
  {
    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();

    sb.append("%%%%\n");
    sb.append("%%%% Declarations and configurations\n");
    sb.append("%%%%\n\n");
    sb.append(":- set_prolog_flag(singleton,off).\n");
    sb.append(":- include('bn_common.psm').\n");
    sb.append(":- dynamic evidence/2.\n\n");
    sb.append(":- set_sw.\n\n");

    sb.append("target(world/0).\n");
    sb.append("target(world/1).\n");
    if (modeZeroComp)
      sb.append("values(bn(Var,Pa),Vs):- range(Var,Pa,Vs).\n\n");
    else
      sb.append("values(bn(Var,_),Vs):- range(Var,Vs).\n\n");
    sb.append("%%%%\n");
    sb.append("%%%% Modeling part for probability computation\n");
    sb.append("%%%% (network specific)\n");
    sb.append("%%%%\n\n");
    sb.append("world(Es):- assert_evidence(Es),!,world.\n\n");
    sb.append("world:- " + headAry[l - 1].toPrismString() + ".\n\n");

    psm.print(sb);
  }

  /** Writes the modeling part to the outoput. */
  public void printPrismBuckets() throws B2PException
  {
    int l = getNumberOfVariables();

    if (headAry == null && headAry[l - 1] == null)
      throw new B2PException("Null headAry");

    StringBuilder sb1 = new StringBuilder();
    StringBuilder sb2 = new StringBuilder();

    for (int q = (l - 1); q >= 0; q--) {
      int i = elimOrder[q];
      Cluster head = headAry[q];

      if (head == null)	throw new B2PException("Null head");

      sb1.append(head.toPrismString());
      sb1.append(":- ");
      sb1.append(head.toPrismString2());
      sb1.append(".\n");

      sb2.append(head.toPrismString2());
      sb2.append(":- ");

      List<Cluster> bucket = bucketList.get(q);
      boolean isFirst = true;
      for (Cluster c : bucket) {
	if (!isFirst) sb2.append(",");
	sb2.append(c.toPrismString());
	isFirst = false;
      }
      sb2.append(".\n");
    }

    sb1.append("\n");

    psm.print(sb1);
    psm.print(sb2);
  }

  /** Writes definitions of <CODE>range/2</CODE> into the output according to the value of modeZeroComp.
   *  Also writes definitions of <CODE>range/3</CODE> If <CODE>modeZeroComp</CODE> is true.
   */
  public void printPrismRanges()
  {
    printPrismRanges2();
    if (modeZeroComp) {
      psm.println();
      printPrismRanges3();
    }
  }

  /** Writes definitions of <CODE>range/2</CODE> into the output <I>without</I> zero compression. */
  private void printPrismRanges2()
  {
    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();
    
    sb.append("\n");
    for (int i = 0; i < l; i++) {
      Variable v = varList.get(i);
      int u = v.getOutcomeSize();
      sb.append("range(x" + i + ",[");
      for (int k = 0; k < u; k++) {
	if (k > 0) sb.append(",");
	sb.append("v" + i + "_" + k);
      }
      sb.append("]).\n");
    }

    psm.print(sb);
  }

  /** Writes definitions of <CODE>range/3</CODE> into the output <I>with</I> zero compression. */
  private void printPrismRanges3()
  {
    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();
    
    for (int i = 0; i < l; i++) {
      Variable v = varList.get(i);
      List<Integer> parents = v.getParentList();

      int[] parSizes = new int[parents.size()];
      int[] parCounters = new int[parents.size()];

      for (int j = 0; j < parents.size(); j++) {
	parSizes[j] = varList.get(parents.get(j)).getOutcomeSize();
	parCounters[j] = 0;
      }

      double[] pAry = v.getTable();
      int numEntries = pAry.length;
      int size = v.getOutcomeSize();
      int numParInstances = numEntries / size;

      for (int row = 0; row < numParInstances; row++) {
	sb.append("range(x" + i + ",[");
	for (int j = 0; j < parents.size(); j++) {
	  if (j > 0) sb.append(",");
	  sb.append("x" + parents.get(j) + "=v" + parents.get(j) + "_" + parCounters[j]);
	}
	sb.append("],[");

	boolean first = true;
	for (int column = 0; column < size; column++) {
	  int k = size * row + column;
	  if (pAry[k] > TINY_PROB) {
	    if (!first) sb.append(",");
	    sb.append("v" + i + "_" + column);
	    first = false;
	  }
	}
	sb.append("]).\n");

	// Increment the indexes:
	for (int j = parents.size() - 1; j >= 0; j--) {
	  parCounters[j]++;
	  if (parCounters[j] >= parSizes[j]) {
	    parCounters[j] = 0;
	  }
	  else break;
	}
      }
    }
    sb.append("\n");

    psm.print(sb);
  }

  /** Writes the definition of the <CODE>cpt</CODE> predicate. */
  public void printPrismCPT()
  {
    StringBuilder sb = new StringBuilder();

    sb.append("\n%%%%\n");
    sb.append("%%%% Modeling part for ");
    sb.append("probability computation (network independent):\n");
    sb.append("%%%%\n\n");
    sb.append("cpt(X,Pa,V):-\n");
    //sb.append("  instanciate_parents(Pa),\n");
    sb.append("  ( evidence(X,V) -> msw(bn(X,Pa),V)\n");
    sb.append("  ; msw(bn(X,Pa),V)\n");
    sb.append("  ).\n\n");

    psm.print(sb);
  }

  /** Gets the total order of variables by topological sorting. */
  public void getTotalOrder()
  {
    int l = getNumberOfVariables();
    VStat[] visited = new VStat[l];

    order = new int[l];
    for (int i = 0; i < l; i++)
      order[i] = -1;

    for (int i = 0; i < l; i++)
      visited[i] = VStat.NEVER;

    int k = 0;
    for (int i = 0; i < l; i++)
      if (visited[i] == VStat.NEVER)
	k = visit(l, i, k, visited);

    for (int i = 0; i < l; i++) {
      Variable v = varList.get(i);
      v.setOrder(order[i]);
    }
  }

  /**
   * Visits each variable in the BN based on the direct adjacency table.
   * This is a depth-first recursive routine for topological sorting.
   */
  private int visit(int l, int i, int k, VStat[] visited)
  {
    visited[i] = VStat.JUST;

    for (int j = 1; j < l; j++) {
      if (adjTab[j][i] <= 0) continue;
      if (visited[j] == VStat.NEVER)
	k = visit(l, j, k, visited);
      else if (visited[j] == VStat.JUST) {
	System.err.println("There is a cycle in the input BN!!");
	System.exit(1);
      }
    }

    order[k] = i;
    visited[i] = VStat.ONCE;

    return (k + 1);
  }

  /** Builds the inverse mapping of <CODE>order[]</CODE>. */
  public void getInverseTotalOrder()
  {
    if (order == null) getTotalOrder();
    int l = getNumberOfVariables();

    invOrder = new int[l];

    for (int m = 0; m < l; m++) {
      invOrder[order[m]] = m;
    }
  }

  /** Gets the primary clusters for all variables. */
  public void getPrimaryCluster()
  {
    int l = getNumberOfVariables();

    primCluster = new int[l];
    primClusterWidth = new int[l];

    for (int i = 0; i < l; i++) {
      primCluster[i] = -1;
      primClusterWidth[i] = 0;
    }

    for (int q = 0; q < l; q++) {
      Cluster head = headAry[q];
      int id = head.getId();
      int width = head.getWidth();
      VArray va = head.getVarAry();

      for (int i = 0; i < va.size(); i++) {
	if ((i == id || va.get(i)) &&
	    (primClusterWidth[i] <= 0 || width < primClusterWidth[i])) {
	  primCluster[i] = q;
	  primClusterWidth[i] = width;
	}
      }
    }

  }

  /**
   * Writes the naive version of the modeling part.  This part will be skipped
   * if we have more than <CODE>MAX_NVAR_NAIVE</CODE> variables.
   */
  public void printPrismNaiveBN()
  {
    int l = getNumberOfVariables();

    if (l > MAX_NVAR_NAIVE) {
      StringBuilder sb = new StringBuilder();

      sb.append("\n%%%% !!\n");
      sb.append("%%%% !! Naive definition of a BN is omitted, since the number of\n");
      sb.append("%%%% !! variables (" + l + ") exceeds the limit (" + MAX_NVAR_NAIVE + ").\n");
      sb.append("%%%% !!\n\n");

      psm.print(sb);

      return;
    }

    StringBuilder sb = new StringBuilder();

    sb.append("\n%%%%\n");
    sb.append("%%%% Naive definition of a BN (similar to sampling model)\n");
    sb.append("%%%%\n\n");

    sb.append("world_n(Es):- assert_evidence(Es),!,world_n0.\n\n");

    sb.append("world_n0:- world_n0(");
    for (int i = 0; i < l; i++) {
      if (i > 0) sb.append(",");
      sb.append("_");
    }
    sb.append(").\n\n");

    sb.append("world_n0(");
    for (int i = 0; i < l; i++) {
      sb.append("X" + i);
      if (i < (l - 1)) sb.append(",");
      if (i % 10 == 9) sb.append("\n        ");
    }
    sb.append("):-\n");
    for (int k = 0; k < l; k++) {
      int i = order[k];
      sb.append("  " + "cpt(x" + i + ",[");
      int q = 0;
      Variable v = varList.get(i);
      for (int j : v.getParentList()) {
	if (q > 0) sb.append(",");
	sb.append("x" + j + "=X" + j);
	q++;
      }
      sb.append("],X" + i + ")");
      if (k < (l - 1))
	sb.append(",\n");
      else 
	sb.append(".\n");
    }
    sb.append("\n");

    psm.print(sb);
  }

  /** Writes the naive version of the modeling part. */
  public void printPrismDistrib()
  {
    if (primCluster == null || primClusterWidth == null)
      getPrimaryCluster();

    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();

    sb.append("\n%%%%\n");
    sb.append("%%%% Utility part (network specific)\n");
    sb.append("%%%%\n\n");
    
    sb.append("j_hprob(Es,SubG,Ps):- ");
    sb.append("chindsight_agg(world(Es),SubG,Ps).\n\n");

    for (int i = 0; i < l; i++) {
      int q = primCluster[i];
      int w = primClusterWidth[i];

      Cluster head = headAry[q];
      VArray va = head.getVarAry();
      int id = head.getId();

      sb.append("j_dist(x" + i + ",Es):- ");
      sb.append("j_hprob(Es,pot_x" + id + "(");

      boolean isFirst = true;
      for (int j = 0; j < va.size(); j++) {
	if (va.get(j) || j == id) {
	  if (!isFirst) sb.append(",");
	  if (j == i) {
	    sb.append("query");
	  }
	  else {
	    sb.append("_");
	  }
	  isFirst = false;
	}
      }

      sb.append("),Ps),!,print_distrib1(x" + i + ",Ps).\n");
    }
    sb.append("\n");

    if (l > MAX_NVAR_NAIVE) {
      sb.append("\n%%%% !!\n");
      sb.append("%%%% !! Naive definition of a BN is omitted, since the number of\n");
      sb.append("%%%% !! variables (" + l + ") exceeds the limit (" + MAX_NVAR_NAIVE + ").\n");
      sb.append("%%%% !!\n\n");
    }
    else {
      sb.append("n_hprob(Es,SubG,Ps):- chindsight_agg(world_n(Es),SubG,Ps).\n\n");
      for (int i = 0; i < l; i++) {
	sb.append("n_dist(x" + i + ",Es):- ");
	sb.append("n_hprob(Es,world_n0(");
	for (int j = 0; j < l; j++) {
	  if (j > 0) sb.append(",");
	  if (i == j)
	    sb.append("query");
	  else
	    sb.append("_");
	}
	sb.append("),Ps),!,");
	sb.append("print_distrib1(x" + i + ",Ps).\n");
      }
    }

    psm.print(sb);
  }

  /** Writes the routine for parameter settings. */
  public void printPrismSetSwitches()
  {
    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();

    sb.append("\n%%%%\n");
    sb.append("%%%% Parameter setting:\n");
    sb.append("%%%%\n\n");

    sb.append("set_sw:-\n");
    for (int i = 0; i < l; i++) {
      sb.append("  set_sw_sub(x" + i + ")");
      if (i == (l - 1))
	sb.append(".");
      else
	sb.append(",");
      sb.append("\n");
    }
    sb.append("\n");

    for (int i = 0; i < l; i++) {
      sb.append("set_sw_sub(x" + i + "):-\n");

      Variable v = varList.get(i);
      List<Integer> parents = v.getParentList();

      int[] parSizes = new int[parents.size()];
      int[] parCounters = new int[parents.size()];

      for (int j = 0; j < parents.size(); j++) {
	parSizes[j] = varList.get(parents.get(j)).getOutcomeSize();
	parCounters[j] = 0;
      }

      double[] pAry = v.getTable();
      int numEntries = pAry.length;
      int size = v.getOutcomeSize();
      int numParInstances = numEntries / size;

      for (int row = 0; row < numParInstances; row++) {
	sb.append("    set_sw(bn(x" + i + ",[");
	for (int j = 0; j < parents.size(); j++) {
	  if (j > 0) sb.append(",");
	  sb.append("x" + parents.get(j) + "=v" + parents.get(j) + "_" + parCounters[j]);
	}
	sb.append("]),[");

	if (modeZeroComp) {
	  boolean first = true;
	  for (int column = 0; column < size; column++) {
	    int k = size * row + column;
	    if (pAry[k] > TINY_PROB) {
	      if (!first) sb.append(",");
	      sb.append(String.format("%.15g",pAry[k]));
	      first = false;
	    }
	  }
	}
	else {
	  for (int column = 0; column < size; column++) {
	    if (column > 0) sb.append(",");
	    int k = size * row + column;
	    sb.append(String.format("%.15g",pAry[k]));
	  }
	}

	if (row < (numParInstances - 1))
	  sb.append("]),\n");
	else
	  sb.append("]).\n");

	// Increment the indexes:
	for (int j = parents.size() - 1; j >= 0; j--) {
	  parCounters[j]++;
	  if (parCounters[j] >= parSizes[j]) {
	    parCounters[j] = 0;
	  }
	  else break;
	}
      }
      
    }

    psm.println(sb);
  }

  /** Writes the routine for sampling. */
  public void printPrismSampler()
  {
    if (order == null) getTotalOrder();
    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();

    sb.append("\n%%%%\n");
    sb.append("%%%% Modeling part for sampling (network specific):\n");
    sb.append("%%%%\n\n");
    sb.append("world_s([");
    for (int i = 0; i < l; i++) {
      sb.append("x" + i + "=X" + i);
      if (i < (l - 1)) sb.append(",");
      if (i % 5 == 4) sb.append("\n         ");
    }
    sb.append("]):-\n");
    for (int k = 0; k < l; k++) {
      int i = order[k];
      sb.append("  msw(bn(x" + i + ",[");
      int q = 0;
      Variable v = varList.get(i);
      for (int j : v.getParentList()) {
	if (q > 0) sb.append(",");
	sb.append("x" + j + "=X" + j);
	q++;
      }
      sb.append("]),X" + i + ")");
      if (k < (l - 1))
	sb.append(",\n");
      else 
	sb.append(".\n");
    }
    sb.append("\n");

    psm.print(sb);
  }

  /**
   * Writes the predicates that relate the names systematically given by
   * the translator to the original ones given in the XML input.
   */
  public void printPrismNames()
  {
    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();

    sb.append("\n%%%%\n");
    sb.append("%%%% Names of the network, variables and their values\n");
    sb.append("%%%%\n\n");

    sb.append("network_name(" + basename + ").\n\n");

    for (int i = 0; i < l; i++) {
      String varName = varList.get(i).getName();
      sb.append("var_name(x" + i + ",'" + varName + "').\n");
    }
    sb.append("\n");
    for (int i = 0; i < l; i++) {
      List<String> outcomeList = varList.get(i).getOutcomeList();
      int k = 0;
      for (String valName : outcomeList) {
	sb.append("val_name(x" + i + ",v" + i + "_" + k + ",'" + valName + "').\n");
	k++;
      }
    }
    sb.append("\n");

    psm.print(sb);
  }

  /** Writes the total order of the variables in the BN. */
  public void printTotalOrder()
  {
    if (order == null) getTotalOrder();
    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();

    sb.append("%%%%\n");
    sb.append("%%%% Topological Order = [");
    for (int i = 0; i < l; i++) {
      if (i > 0) sb.append(",");
      sb.append(varList.get(i).getOrder());
    }
    sb.append("]\n");
    sb.append("%%%%\n\n");

    psm.print(sb);
  }

  /** Writes the variables in the BN. */
  public void printVariables()
  {
    StringBuilder sb = new StringBuilder();
    
    sb.append("%%%%\n");
    sb.append("%%%% Variables:\n");
    sb.append("%%%%\n\n");

    for (Variable var : varList) {
      sb.append("%% " + var.getId() + ":" + var.getName());
      sb.append("\n");
    }
    sb.append("\n");

    psm.print(sb);
  }

  /** Computes the evidence levels for all variables in the BN. */
  public void getEvidLevel()
  {
    int l = getNumberOfVariables();
    evidLevel = new int[l];

    List<Integer> evidLevelList = new ArrayList<Integer>();

    int s0 = l - (l / NUM_EVID_LEVEL) * (NUM_EVID_LEVEL - 1);

    for (int s = 0; s < s0; s++)
      evidLevelList.add(0);

    for (int t = 1; t < NUM_EVID_LEVEL; t++)
      for (int s = 0; s < (l / NUM_EVID_LEVEL); s++)
	evidLevelList.add(t);

    Collections.shuffle(evidLevelList);

    for (int i = 0; i < l; i++)
      evidLevel[i] = evidLevelList.get(i);
  }

  /** Writes the evidences. */
  public void printPsmEvid()
  {
    if (!hasEvid) return;

    int l = getNumberOfVariables();

    evidPsm.print("evidence([");
    boolean isFirst = true;

    for (int i = 0; i < l; i++) {
      Variable v = varList.get(i);
      if (v.getEvidence() >= 0) {
	if (!isFirst) evidPsm.print(",");
	evidPsm.print("x"+ i + "=v" + i + "_" + v.getEvidence());
	isFirst = false;
      }
    }

    evidPsm.println("]).");
  }

  /**
   * Writes the random evidences.  To handle the case where zero-compression is enabled,
   * we should make a sampling following the topological order to take the context-specificity
   * into account.
   */
  public void printPsmRandomEvid()
  {
    int l = getNumberOfVariables();

    StringBuilder sb = new StringBuilder();

    sb.append("%% Name of network: " + basename + "\n");
    sb.append("%% Number of evidence levels:\n");
    sb.append("evid_level(" + NUM_EVID_LEVEL + ").\n\n");

    psm.print(sb);

    Random rand = new Random();
    int[] sampled = new int[l];

    for (int q = 0; q < l; q++) {
      int i = order[q];
      Variable v = varList.get(i);
      List<Integer> parents = v.getParentList();

      double[] tab = v.getTable();
      int outSize = v.getOutcomeSize();

      int row = 0;
      for (int j = 0; j < parents.size(); j++) {
	int o = sampled[parents.get(j)];
	for (int m = (j + 1); m < parents.size(); m++)
	  o *= varList.get(parents.get(m)).getOutcomeSize();
	row += o;
      }

      double[] probs = new double[outSize];
      for (int k = 0; k < outSize; k++)
	probs[k] = tab[row * outSize + k];

      double r = rand.nextDouble();
      double sum = 0.0;
      for (int k = 0; k < outSize; k++) {
	sum += probs[k];
	if (sum > r) {
	  sampled[i] = k;
	  break;
	}
      }
    }

    for (int t = 0; t < NUM_EVID_LEVEL; t++) {
      randEvidPsm[t].print("evidence([");
      boolean isFirst = true;
      for (int i = 0; i < l; i++) {
	if (evidLevel[i] >= (NUM_EVID_LEVEL - t)) {
	  if (!isFirst) randEvidPsm[t].print(",");
	  randEvidPsm[t].print("x"+ i + "=v" + i + "_" + sampled[i]);
	  isFirst = false;
	}
      }
      randEvidPsm[t].println("]).");
    }
  }

  /** Returns a string representation. */
  public String toString()
  {
    StringBuilder sb = new StringBuilder();

    for (Variable var : varList) {
      sb.append(varMap.get(var.getName()) + ":");
      sb.append(var);
      sb.append("\n");
    }

    return sb.toString();
  }
}
