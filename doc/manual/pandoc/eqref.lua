-- pandoc Lua filter used by Make_html.sh
--
-- pandoc cannot number equations, so it renders \ref{eq:...} as the label
-- itself ("[eq:adam]") linked to a missing anchor.  References to labels
-- defined inside formulas are replaced with the inline formula \ref{...},
-- which MathJax resolves to the equation number (tags: 'ams', configured
-- in header.html).

local labels = {}

local function collect_labels(math)
  for label in math.text:gmatch('\\label%s*{([^}]*)}') do
    labels[label] = true
  end
end

local function replace_ref(link)
  local ref_type = link.attributes['reference-type']
  local label = link.target:gsub('^#', '')
  if (ref_type == 'ref' or ref_type == 'eqref') and labels[label] then
    return pandoc.Math('InlineMath', '\\' .. ref_type .. '{' .. label .. '}')
  end
end

function Pandoc(doc)
  doc:walk({Math = collect_labels})
  return doc:walk({Link = replace_ref})
end
