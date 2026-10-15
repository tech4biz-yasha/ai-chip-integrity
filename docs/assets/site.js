(function () {
  // Menu on small screens
  var menu = document.querySelector('.menu'), nav = document.getElementById('nav');
  if (menu && nav) menu.addEventListener('click', function () {
    var open = nav.classList.toggle('open'); menu.setAttribute('aria-expanded', open ? 'true' : 'false');
  });
  // Links into the old one-page site (chipintegrity.org/#results) now live on their own pages
  var moved = {silent: 'learn', why: 'learn', glossary: 'learn', method: 'method', verdicts: 'method', selftest: 'method',
               limits: 'method', results: 'results', run: 'run', roadmap: 'run', research: 'about', about: 'about'};
  var h = location.hash.slice(1);
  if (document.body.getAttribute('data-page') === 'index' && moved[h]) { location.replace(moved[h] + '.html#' + h); return; }
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  // Numbers count up to the value already written in the page
  function count(el) {
    var to = el.getAttribute('data-to'); if (!to || reduce) return;
    var target = parseFloat(to.replace(/,/g, '')); if (!isFinite(target) || target === 0) return;
    var dec = (to.split('.')[1] || '').length, text = el.textContent, start = null;
    function step(ts) {
      if (start === null) start = ts;
      var k = Math.min(1, (ts - start) / 1300), v = target * (1 - Math.pow(1 - k, 3));
      el.textContent = text.replace(to, v.toLocaleString('en-US', {minimumFractionDigits: dec, maximumFractionDigits: dec}));
      if (k < 1) requestAnimationFrame(step); else el.textContent = text;
    }
    requestAnimationFrame(step);
  }
  // Sections ease in as they reach the screen
  var els = [].slice.call(document.querySelectorAll('.reveal'));
  if ('IntersectionObserver' in window && !reduce) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        e.target.classList.add('in'); [].forEach.call(e.target.querySelectorAll('[data-to]'), count); io.unobserve(e.target);
      });
    }, {threshold: 0.12});
    els.forEach(function (el) { io.observe(el); });
  } else els.forEach(function (el) { el.classList.add('in'); });
  // The contents list follows the section being read
  var toc = [].slice.call(document.querySelectorAll('.toc a[href^="#"]'));
  if (toc.length) {
    var secs = toc.map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); });
    var on = function () {
      var y = window.scrollY + 140, cur = null;
      secs.forEach(function (s) { if (s && s.offsetTop <= y) cur = s; });
      toc.forEach(function (a) { a.classList.toggle('on', !!cur && a.getAttribute('href') === '#' + cur.id); });
    };
    window.addEventListener('scroll', on, {passive: true}); on();
  }
})();
