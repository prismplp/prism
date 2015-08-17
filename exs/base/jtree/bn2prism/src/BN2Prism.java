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

import java.io.*;
import javax.xml.parsers.*;
import org.w3c.dom.*;

/**
 * The main class of the translator.  The name of the input network file should be
 * "BASENAME.xml".
 * <PRE>
 * [Usage] java BN2Prism [-f FORMAT][-o][-t][-r][-e][-z][-n][-v][-h] BASENAME
 * </PRE>
 *
 * <P>
 * Command line options:
 * <UL>
 * <LI><CODE>-f FORMAT</CODE>: the input format is FORMAT (xmlbif or xbif; default: xmlbif). </LI>
 * <LI><CODE>-o</CODE>: the translator reads an external file (named BASENAME.num) that specifies
 * the elimination order. </LI>
 * <LI><CODE>-t</CODE>: the translator reads transposed CPTs.</LI>
 * <LI><CODE>-r</CODE>: the translator assigns random probabilities to the parameters.</LI>
 * <LI><CODE>-e</CODE>: the translator generates auxiliary files that specify the random
 * evidences.</LI>
 * <LI><CODE>-z</CODE>: the translator makes zero-compression of CPTs.</LI>
 * <LI><CODE>-n</CODE>: the translator normalizes the entries in CPTs.</LI>
 * <LI><CODE>-v</CODE>: the translator outputs detailed messages as comments.</LI>
 * <LI><CODE>-V</CODE>: the translator outputs the version.</LI>
 * <LI><CODE>-h</CODE>: a brief help is displayed.</LI>
 * </UL>
 * </P>
 *
 * <P>
 * FORMAT is either <CODE>xmlbif</CODE> (for XMLBIF) or <CODE>xbif</CODE> (for XBIF).
 * The default is XMLBIF.  XMLBIF used here is basically based on XMLBIF 0.3,
 * but the entries in each CPT have a different order from those in XMLBIF 0.3
 * (actually, the order is same as that of XMLBIF 0.5).  We can switch the entry
 * order by the <CODE>-t</CODE> option. XBIF, on the other hand, is the format
 * adopted in UAI-06 Evaluation Workshop.  When XMLBIF is specified, a separate
 * evidence file (in so-called XMLBIF-EVIDENCE format) "BASENAME_evid.xml" is
 * required to take evidences into account.
 * </P>
 *
 * <P>
 * After running the command, the network spacification is translated into a PRISM
 * file named "BASENAME.psm"  The evidences (if given) are translated into a separate
 * file named "BASENAME_evid.psm".  If the <CODE>-e</CODE> option is specified, the
 * translator generates a couple of files each of which contains randomly generated
 * evidences.
 * </P>
 *
 * <P>
 * For example, the ALARM network is specified in "alarm.xml" in the XMLBIF format,
 * we can get the junction-tree PRISM program for the ALARM network by running:
 * <PRE>
 * java BN2Prism -f xmlbif alarm
 * </PRE>
 * (we can omit "-f xmlbif" since it is the default).  Please note here that we need
 * to prepare the evidence file named "alarm_evid.xml" unless we assign the random
 * probabilities to CPTs (with the <CODE>-r</CODE> option) or have no evidences.
 * Even if alarm.xml have CPTs with the transposed form, we can force the translator
 * to accept such CPTs by specifying the <CODE>-t</CODE> option:
 * <PRE>
 * java BN2Prism -f xmlbif -t alarm
 * </PRE>
 * On the other hand, when the network is provided in the XBIF format, the
 * <CODE>-f</CODE> option is mandatory.  In this case, the evidences are buried
 * in the network specification, so no separate evidence file is required.
 * <PRE>
 * java BN2Prism -f xbif alarm
 * </PRE>
 * </P>
 */
public class BN2Prism
{
  /** Message on usage */
  public final static String USAGE
    = "[Usage] java BN2Prism [-f FORMAT][-o][-t][-r][-e][-z][-n][-v][-V][-h] BASENAME\n";

  /** String representation of the version number */
  public final static String VERSION = "1.3";

  /** Settings for the translator. */
  private BNSetting setting;

  /** The default setting will be set. */
  public BN2Prism() {
    setting = new BNSetting(VERSION);
  }

  /** Returns the current setting of the translator. */
  public BNSetting getSetting() {
    return setting;
  }

  /**
   * Main method.
   * @param args An array of strings which contains command arguments.
   *             It is assumed that <code>args[0]</code> is the
   *             filename of the input file.
   */
  public static void main(String[] args)
  {
    try {
      long total_s_time = System.currentTimeMillis();

      BN2Prism t = new BN2Prism();

      t.parseArgs(args);
      
      long read_s_time = System.currentTimeMillis();
      BayesNet bn = t.buildBN();
      long read_e_time = System.currentTimeMillis();

      //System.err.println(bn);
      if (t.getSetting().getModeVerbose()) bn.printVariables();

      //// Output a Prism program:

      long trans_s_time = System.currentTimeMillis();

      // Preparations:
      bn.getAdjacencyTable();
      bn.getUndirectedAdjacencyTable();

      if (t.getSetting().getModeVerbose()) bn.printAdjacencyTable();

      bn.getElimOrder();
      bn.getTotalOrder();
      bn.getInverseTotalOrder();
      if (t.getSetting().getModeVerbose()) bn.printTotalOrder();

      bn.buildInitialBuckets();
      if (t.getSetting().getModeVerbose()) bn.printInitialBuckets();
      
      bn.growBuckets();
      if (t.getSetting().getModeVerbose()) bn.printFinalBuckets();

      bn.buildModes();
      bn.sortBuckets();
      if (t.getSetting().getModeVerbose()) bn.printModedBuckets();

      bn.printPrismVersion();

      bn.printPrismModes();
      bn.printPrismDecls();
      bn.printPrismBuckets();
      bn.printPrismRanges();

      bn.printPrismCPT();
      bn.printPrismSampler();
      bn.printPrismNaiveBN();

      bn.getPrimaryCluster();
      bn.printPrismDistrib();
      bn.printPrismNames();
      bn.printPrismSetSwitches();

      long trans_e_time = System.currentTimeMillis();

      //// Output evidence file(s) specified in the input:
      bn.printPsmEvid();

      //// Output random evidence files:
      if (t.getSetting().getModeEvid()) {
	bn.getEvidLevel();
	bn.printPsmRandomEvid();
      }

      bn.closeStreams();

      long total_e_time = System.currentTimeMillis();

      System.out.println("Time (read)  = " + (read_e_time - read_s_time) + " [msec]");
      System.out.println("Time (trans) = " + (trans_e_time - trans_s_time) + " [msec]");
      System.out.println("Time (total) = " + (total_e_time - total_s_time) + " [msec]");
    }
    catch (Exception e) {
      e.printStackTrace();
    }
  }

  private void parseArgs(String[] args) throws B2PException
  {
    for (int i = 0; i < args.length; i++) {
      if (args[i].equals("-f")) {
	if (i >= (args.length - 1)) {
          System.err.println("Unexpected argument(s)");
          System.exit(1);
	}
	else {
          if (args[i+1].equals("xmlbif")) {
            setting.setFormat(Format.XMLBIF);
          }
          else if (args[i+1].equals("xbif")){
            setting.setFormat(Format.XBIF);
          }
          else {
            System.err.println("Unexpected file format: " + args[i+1]);
            System.exit(1);
          }
	}
	i++;
      }
      else if (args[i].equals("-o")) {
        setting.setHasExtElimOrder(true);
      }
      else if (args[i].equals("-t")) {
        setting.setHasTrCPT(true);
      }
      else if (args[i].equals("-r")) {
        setting.setModeRandom(true);
      }
      else if (args[i].equals("-e")) {
        setting.setModeEvid(true);
      }
      else if (args[i].equals("-z")) {
	setting.setModeZeroComp(true);
      }
      else if (args[i].equals("-n")) {
	setting.setModeNormalize(true);
      }
      else if (args[i].equals("-v")) {
        setting.setModeVerbose(true);
      }
      else if (args[i].equals("-V")) {
	System.out.println(VERSION);
	System.exit(0);
      }
      else if (args[i].equals("-h")) {
	System.out.println(USAGE);
	System.exit(0);
      }
      else if (args[i].startsWith("-")) {
	System.err.println("Unknown option flag: " + args[i]);
	System.exit(1);
      }
      else {
	setting.setBaseName(args[i]);
      }
    }

    if (setting.getBaseName() == null)
      throw new B2PException("Can't find a target name");
  }

  /**
   * Parse the input XMLBIF file, and store the information to
   * the intermediate data structure.
   */
  public BayesNet buildBN()
  {
    String netFile  = setting.getBaseName() + ".xml";
    String netFile2 = setting.getBaseName() + ".xbif";
    String evidFile = setting.getBaseName() + "_evid.xml";

    BayesNet bn = null;

    try {
      // Create a factory for a DOM parser:
      DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
      
      // Currently we do not make validations:
      //dbf.setValidating(true);

      // Create a DOM parser:
      DocumentBuilder db = dbf.newDocumentBuilder();
      db.setErrorHandler(new B2PHandler());

      // Parse the network XML file:
      Document document = null;
      if (setting.getFormat() == Format.XBIF &&
	  !(new File(netFile)).exists() && (new File(netFile2)).exists())
        document = db.parse(new FileInputStream(netFile2));
      else
        document = db.parse(new FileInputStream(netFile));

      // Parse the evidence XML file (XMLBIF only):
      Document documentEvid = null;
      if (setting.getFormat() == Format.XMLBIF &&
	  (new File(evidFile)).exists())
        documentEvid = db.parse(new FileInputStream(evidFile));

      // Print out the contents in DOM:
      //printDocument(document);

      // Get the intermediate data structure from DOM:
      bn = new BayesNet(setting, document, documentEvid);
    }
    catch (Exception e) {
      //e.printStackTrace();
      System.err.println(e.getMessage());
      System.exit(1);
    }

    return bn;
  }

  /**
   * Display the contents in a DOM tree.
   * @param document DOM tree to be displayed.
   */
  private void printDocument(Document document)
  {
    Element root = document.getDocumentElement();
    
    System.out.println("---- Result of printDocument() ----");
    traverse(root,0);
  }
  
  /**
   * Display the contents of a DOM tree recursively.
   * @param n the node we are visiting at.
   * @param indent the width of indentation.
   */
  private void traverse(Node n, int indent)
  {
    String s = getNodeString(n);
    if (s.length() > 0) {
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < indent; i++)
	sb.append("  ");
      sb.append(s);
      System.out.println(sb);
    }
    
    for (Node ch = n.getFirstChild(); ch != null; ch = ch.getNextSibling())
      traverse(ch,indent+1);
  }

  /**
   * Get the string which denotes the contents of a node.
   * @param n a node in DOM.
   * @return the string which denotes the contents of the node.
   */
  public String getNodeString(Node n)
  {
    String s = null;
    
    switch (n.getNodeType()) {
    case Node.ELEMENT_NODE:
      s = n.getNodeName();
      break;
    case Node.TEXT_NODE:
      // Replace consective space symbols including new line symbols
      // by one space symbol, and delete the spaces at beginning and
      // and ending:
      s = n.getTextContent().replaceAll("\\s+"," ");
      s = s.trim();
      break;
    default:
      s = "";
    }

    return s;
  }

}
