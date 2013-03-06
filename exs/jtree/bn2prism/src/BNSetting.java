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
 * A setting for the translator.
 */
public class BNSetting
{
  /** Basename of the input file. */
  private String basename;

  /** Version number of the translator. */
  private String version;

  /** Format of the XML input. */
  private Format format;

  /**
   * Flag indicating whether we have an external file (BASENAME.num) which
   * specifies the elimination order.
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

  /** Several default settings are also made. */
  public BNSetting(String v) {
    version = v;
    basename = null;
    format = Format.XMLBIF;

    hasExtElimOrder = false;
    hasTrCPT        = false;
    modeZeroComp    = false;
    modeRandom      = false;
    modeEvid        = false;
    modeNormalize   = false;
    modeVerbose     = false;
  }

  public void setBaseName(String n) { basename = n; }
  public void setFormat(Format f)   { format = f;   }

  public void setHasExtElimOrder(boolean b) { hasExtElimOrder = b; }
  public void setHasTrCPT(boolean b)        { hasTrCPT        = b; }
  public void setModeZeroComp(boolean b)    { modeZeroComp    = b; }
  public void setModeRandom(boolean b)      { modeRandom      = b; }
  public void setModeEvid(boolean b)        { modeEvid        = b; }
  public void setModeNormalize(boolean b)   { modeNormalize   = b; }
  public void setModeVerbose(boolean b)     { modeVerbose     = b; }

  public String getBaseName() { return basename; }
  public String getVersion()  { return version;  }
  public Format getFormat()   { return format;   }

  public boolean getHasExtElimOrder() { return hasExtElimOrder; }
  public boolean getHasTrCPT()        { return hasTrCPT;        }
  public boolean getModeZeroComp()    { return modeZeroComp;    }
  public boolean getModeRandom()      { return modeRandom;      }
  public boolean getModeEvid()        { return modeEvid;        }
  public boolean getModeNormalize()   { return modeNormalize;   }
  public boolean getModeVerbose()     { return modeVerbose;     }
}
