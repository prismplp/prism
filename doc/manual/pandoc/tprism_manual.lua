-- pandoc Lua filter used by Make_tprism_html.sh
--
-- - pandoc does not number captions in HTML, while the text refers to
--   "Figure 9.1" and "Table 6.1".  The captions of figures and tables are
--   numbered per chapter as in LaTeX, where a figure or table without a
--   caption gets no number.
-- - In LaTeX, the spaces after \tt and \it are skipped, but pandoc keeps
--   them at the start of the group when the group continues on the next
--   line ({\tt<newline>tprism}), which shows as a double space in HTML.

traverse = 'topdown'

local chapter, nfig, ntab = 0, 0, 0

local function number_caption(caption, name, n)
  local first = caption.long[1]
  if first and (first.t == 'Plain' or first.t == 'Para') then
    first.content:insert(1, pandoc.Space())
    first.content:insert(1, pandoc.Str(name .. ' ' .. chapter .. '.' .. n .. ':'))
  end
end

function Header(h)
  if h.level == 1 and not h.classes:includes('unnumbered') then
    chapter, nfig, ntab = chapter + 1, 0, 0
  end
end

function Figure(fig)
  if #fig.caption.long > 0 then
    nfig = nfig + 1
    number_caption(fig.caption, 'Figure', nfig)
    return fig
  end
end

function Table(tbl)
  if #tbl.caption.long > 0 then
    ntab = ntab + 1
    number_caption(tbl.caption, 'Table', ntab)
    return tbl
  end
end

function Span(span)
  local first = span.content[1]
  if span.identifier == '' and first and first.t == 'Code' and first.text:match('^%s') then
    first.text = first.text:gsub('^%s+', '')
    if first.text == '' then
      span.content:remove(1)
    end
    return span
  end
end
