// Turns GitHub-style callouts ("> [!NOTE]", "> [!TIP]", ...) into styled
// boxes on the GitHub Pages site. GitHub draws these itself, but Jekyll's
// Markdown renderer leaves the marker as literal text, so without this the
// site shows "[!NOTE]" at the start of a plain quote.
(function () {
  "use strict";

  var TYPES = {
    NOTE: ["ℹ️", "Note"],
    TIP: ["💡", "Tip"],
    IMPORTANT: ["❗", "Important"],
    WARNING: ["⚠️", "Warning"],
    CAUTION: ["🛑", "Caution"]
  };
  var MARKER = /^\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*/;

  function convert() {
    var quotes = document.querySelectorAll(".wrapper > section blockquote");
    Array.prototype.forEach.call(quotes, function (quote) {
      var first = quote.firstElementChild;
      if (!first || first.tagName !== "P") return;
      var text = first.firstChild;
      if (!text || text.nodeType !== Node.TEXT_NODE) return;
      var match = text.nodeValue.match(MARKER);
      if (!match) return;

      text.nodeValue = text.nodeValue.slice(match[0].length);
      // The marker had a paragraph of its own: drop the now-empty paragraph.
      if (!first.textContent.trim() && !first.children.length) quote.removeChild(first);

      var type = TYPES[match[1]];
      var title = document.createElement("p");
      title.className = "callout-title";
      title.textContent = type[0] + " " + type[1];
      quote.insertBefore(title, quote.firstChild);
      quote.classList.add("callout", "callout-" + match[1].toLowerCase());
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", convert);
  } else {
    convert();
  }
})();
