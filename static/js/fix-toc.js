// Copyright (c) 2017 Yihui Xie
// & 2018 Vincent Tam (under MIT)
// & 2020 Roman Ring (under MIT)
function fix_toc(root) {
  if (!root) return;

  var ul = root.querySelector('ul');
  if (!ul) return;

  if (ul.childElementCount == 1) {
    var li = ul.firstElementChild;
    if (li.tagName !== 'LI') return;
    fix_toc(ul);
    ul.outerHTML = li.innerHTML;
  } else if (ul.childElementCount > 1) {
    for (i = 0; i < ul.childElementCount; i++) {
      // fix_toc(ul.children[i]); 
      var ul2 = ul.children[i].querySelector('ul');
      if (ul2 && ul2.childElementCount == 1) {
        var li2 = ul2.firstElementChild;
        ul2.outerHTML = li2.innerHTML;
      }
    }
  }
}


(function() {
  var toc = document.getElementById('TableOfContents');
  fix_toc(toc);
})();
