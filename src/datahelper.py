import torch
import re
import torch
class BatchWrapper:
      def __init__(self, dl, x_var, y_vars):
            self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x &amp;amp;amp;amp;lt;g class="gr_ gr_3178 gr-alert gr_spell gr_inline_cards gr_disable_anim_appear ContextualSpelling ins-del" id="3178" data-gr-id="3178"&amp;amp;amp;amp;gt;and y&amp;amp;amp;amp;lt;/g&amp;amp;amp;amp;gt;
  
      def __iter__(self):
            for batch in self.dl:
                  x = getattr(batch, self.x_var) # we assume only one input in this wrapper
  
                  if self.y_vars:
                        y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1)
                  else:
                        y = torch.zeros((1))

                  yield (x, y.view(-1))
  
      def __len__(self):
            return len(self.dl)

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"<br />",r" ",s)
    s = re.sub(r'(\W)(?=\1)', '', s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    
    return s
