// "On this page" navigation for the GitHub Pages site.
// Lists the page's ## and ### headings beside the content on wide screens,
// or in a collapsed box at the top on narrow ones, and highlights the
// section being read. Pages with fewer than two such headings are left as
// they are, and so is every page when JavaScript is off.
(function () {
  "use strict";

  var TOP_BAR = 100; // px below the top of the window that counts as "reading"

  function slug(text) {
    return text.toLowerCase().trim().replace(/[^\w\s-]/g, "").replace(/\s+/g, "-");
  }

  function makeList(headings) {
    var root = document.createElement("ul");
    var sub = null;
    headings.forEach(function (h) {
      var li = document.createElement("li");
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      a.setAttribute("data-target", h.id);
      li.appendChild(a);
      if (h.tagName === "H3" && root.lastElementChild) {
        if (!sub || sub.parentNode !== root.lastElementChild) {
          sub = document.createElement("ul");
          root.lastElementChild.appendChild(sub);
        }
        sub.appendChild(li);
      } else {
        root.appendChild(li);
        sub = null;
      }
    });
    return root;
  }

  function build() {
    var content = document.querySelector(".wrapper > section");
    if (!content) return;

    // Skip the site title block, and headings inside collapsed sections
    // (jumping to hidden content would land nowhere visible).
    var headings = Array.prototype.filter.call(content.querySelectorAll("h2, h3"), function (h) {
      return !h.closest("#title") && !h.closest("details");
    });
    if (headings.length < 2) return;

    headings.forEach(function (h) {
      if (h.id) return;
      var base = slug(h.textContent) || "section";
      var id = base;
      for (var n = 2; document.getElementById(id); n++) id = base + "-" + n;
      h.id = id;
    });

    var side = document.createElement("nav");
    side.className = "page-nav page-nav-side";
    side.setAttribute("aria-label", "On this page");
    var title = document.createElement("p");
    title.className = "page-nav-title";
    title.textContent = "On this page";
    side.appendChild(title);
    side.appendChild(makeList(headings));
    document.body.appendChild(side);

    var inline = document.createElement("details");
    inline.className = "page-nav page-nav-inline";
    var summary = document.createElement("summary");
    summary.textContent = "On this page";
    inline.appendChild(summary);
    inline.appendChild(makeList(headings));
    var titleBlock = content.querySelector("#title");
    content.insertBefore(inline, titleBlock ? titleBlock.nextSibling : content.firstChild);
    inline.addEventListener("click", function (e) {
      if (e.target.tagName === "A") inline.open = false;
    });

    var links = Array.prototype.slice.call(document.querySelectorAll(".page-nav a"));
    function update() {
      var current = headings[0].id;
      var atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2;
      if (atBottom) {
        current = headings[headings.length - 1].id;
      } else {
        for (var i = 0; i < headings.length; i++) {
          if (headings[i].getBoundingClientRect().top - TOP_BAR > 0) break;
          current = headings[i].id;
        }
      }
      links.forEach(function (a) {
        var on = a.getAttribute("data-target") === current;
        a.classList.toggle("is-active", on);
        if (on) a.setAttribute("aria-current", "location");
        else a.removeAttribute("aria-current");
      });
    }

    var pending = false;
    window.addEventListener("scroll", function () {
      if (pending) return;
      pending = true;
      window.requestAnimationFrame(function () {
        pending = false;
        update();
      });
    }, { passive: true });
    window.addEventListener("resize", update);
    update();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();
