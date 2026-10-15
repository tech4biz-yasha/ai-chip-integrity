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

// Teaching tool 1: flip the bits of a number
(function () {
  var box = document.getElementById('bx'); if (!box) return;
  var F = {fp32: {e: 8, m: 23}, bf16: {e: 8, m: 7}, fp16: {e: 5, m: 10}};
  var buf = new DataView(new ArrayBuffer(4));
  function toBits32(x) { buf.setFloat32(0, x); return buf.getUint32(0) >>> 0; }
  function fromBits32(b) { buf.setUint32(0, b >>> 0); return buf.getFloat32(0); }
  function encode(x, fmt) {                       // the value rounded to the nearest number in the format, as bits
    var b = toBits32(x);
    if (fmt === 'fp32') return b;
    if (fmt === 'bf16') { if (x !== x) return 0x7fc0; return ((b + 0x7fff + ((b >>> 16) & 1)) >>> 16) & 0xffff; }
    var s = (b >>> 16) & 0x8000, a = Math.abs(x);
    if (x !== x) return 0x7e00;
    if (a >= 65520) return s | 0x7c00;            // rounds to infinity
    if (a < Math.pow(2, -24) / 2) return s;
    var e = Math.floor(Math.log2(a)); if (e < -14) e = -14;
    var q = Math.round(a / Math.pow(2, e - 10));  // whole number of steps of the last place
    if (e === -14 && q < 1024) return s | q;      // subnormal
    if (q === 2048) { q = 1024; e += 1; }
    if (e > 15) return s | 0x7c00;
    return s | ((e + 15) << 10) | (q - 1024);
  }
  function decode(bits, fmt) {
    if (fmt === 'fp32') return fromBits32(bits);
    if (fmt === 'bf16') return fromBits32(bits << 16);
    var s = bits & 0x8000 ? -1 : 1, e = (bits >>> 10) & 31, m = bits & 1023;
    if (e === 31) return m ? NaN : s * Infinity;
    if (e === 0) return s * m * Math.pow(2, -24);
    return s * (1 + m / 1024) * Math.pow(2, e - 15);
  }
  var fmtSel = document.getElementById('bxfmt'), val = document.getElementById('bxval'), bitsEl = document.getElementById('bxbits');
  var start = 0, cur = 0, fmt = 'fp32', last = null;
  function show(x) {
    if (x !== x) return 'not a number';
    if (!isFinite(x)) return x > 0 ? 'infinity' : '-infinity';
    var a = Math.abs(x); return (a !== 0 && (a >= 1e7 || a < 1e-4)) ? x.toExponential(6) : String(+x.toPrecision(9));
  }
  function paint() {
    var f = F[fmt], n = 1 + f.e + f.m, x0 = decode(start, fmt), x = decode(cur, fmt);
    [].forEach.call(bitsEl.querySelectorAll('button'), function (btn) {
      var i = +btn.getAttribute('data-i'), on = (cur >>> i) & 1;
      btn.textContent = on; btn.setAttribute('aria-pressed', ((start ^ cur) >>> i) & 1 ? 'true' : 'false');
    });
    document.getElementById('bxnow').textContent = show(x);
    var d = x - x0, same = cur === start;
    document.getElementById('bxdelta').textContent = same ? 'none' : (isFinite(d) ? show(d) : 'unbounded');
    document.getElementById('bxrel').textContent = same ? 'none' : (x0 === 0 ? 'from zero' : (isFinite(d) ? (Math.abs(d / x0)).toExponential(2) : 'unbounded'));
    var say = 'Click a bit to flip it.';
    if (!same && last !== null) {
      var field = last === n - 1 ? 's' : (last >= f.m ? 'e' : 'm');
      if (!isFinite(x)) say = 'The value is now ' + show(x) + '. Any check catches that at once.';
      else if (field === 's') say = 'The sign bit turns the answer negative or positive. The reference check catches that, unless the value was zero, where only the bit-for-bit repeat check sees it.';
      else if (field === 'e') say = 'An exponent bit multiplies or divides the value by a power of two. No proven bound allows that much, so the reference check catches it.';
      else say = 'Mantissa bit ' + last + ' moves the value by about ' + Math.pow(2, last - f.m).toExponential(1) + ' of itself. Low bits like this can hide inside the rounding a long sum is allowed, which is why every run is also compared bit for bit with the first.';
    }
    document.getElementById('bxsay').textContent = say;
  }
  function build() {
    var f = F[fmt], n = 1 + f.e + f.m; bitsEl.innerHTML = '';
    for (var i = n - 1; i >= 0; i--) {
      var cls = i === n - 1 ? 's' : (i >= f.m ? 'e' : 'm');
      var b = document.createElement('button'); b.type = 'button'; b.className = cls; b.setAttribute('data-i', i);
      b.setAttribute('aria-label', 'bit ' + i + ', ' + (cls === 's' ? 'sign' : cls === 'e' ? 'exponent' : 'mantissa'));
      b.addEventListener('click', function () { var k = +this.getAttribute('data-i'); cur = (cur ^ (1 << k)) >>> 0; if (fmt !== 'fp32') cur &= 0xffff; last = k; paint(); });
      bitsEl.appendChild(b);
      if (i === n - 1 || i === f.m) { var g = document.createElement('span'); g.className = 'gap'; bitsEl.appendChild(g); }
    }
  }
  function reset() { fmt = fmtSel.value; var x = parseFloat(val.value); if (!isFinite(x)) x = 6; start = cur = encode(x, fmt); last = null; build(); paint(); }
  fmtSel.addEventListener('change', reset); val.addEventListener('change', reset);
  document.getElementById('bxreset').addEventListener('click', reset);
  reset();
})();
// Teaching tool 2: the rounding allowance of a long FP32 sum, against the size of each mantissa bit flip
(function () {
  var box = document.getElementById('bud'); if (!box) return;
  var slider = document.getElementById('budk'), out = document.getElementById('budkv'), bars = document.getElementById('budbars');
  var u = Math.pow(2, -24), lo = -24, hi = 0;           // the chart spans relative sizes 2^-24 to 2^0
  function h(log2v) { return Math.max(0, Math.min(1, (log2v - lo) / (hi - lo))) * 100; }
  for (var b = 0; b < 23; b++) { var d = document.createElement('div'); d.className = 'b'; d.style.height = h(b - 23) + '%'; bars.appendChild(d); }
  var line = document.createElement('div'); line.className = 'line'; bars.appendChild(line);
  function paint() {
    var K = Math.pow(2, +slider.value), Ku = K * u, g = Ku / (1 - Ku), lg = Math.log2(g), hidden = 0;
    out.textContent = K.toLocaleString('en-US');
    [].forEach.call(bars.querySelectorAll('.b'), function (el, i) { var hide = (i - 23) <= lg; el.classList.toggle('hide', hide); if (hide) hidden++; });
    line.style.bottom = h(lg) + '%';
    document.getElementById('budsay').textContent = 'With ' + K.toLocaleString('en-US') + ' terms the allowance is about ' + g.toExponential(1) +
      ' of the sum. The lowest ' + hidden + ' of the 23 mantissa bits could flip and stay inside it (green). A flip in any higher bit, the exponent or the sign would be caught by the reference check.';
  }
  slider.addEventListener('input', paint); paint();
})();

// Results: compare two chips, and filter the tables
(function () {
  var data = document.getElementById('chipdata'), out = document.getElementById('cmpout'); if (!data || !out) return;
  var chips = JSON.parse(data.textContent), a = document.getElementById('cmpa'), b = document.getElementById('cmpb');
  chips.forEach(function (c, i) { [a, b].forEach(function (sel) { var o = document.createElement('option'); o.value = i; o.textContent = c.name; sel.appendChild(o); }); });
  b.value = chips.length > 1 ? 1 : 0;
  function esc(t) { var d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
  function cell(r) { return r == null ? '<div class="none">not measured</div>' : '<div class="cell"><div class="track"><div class="fill" style="width:' + Math.min(100, r * 100).toFixed(1) + '%"></div><div class="limitline"></div></div><div class="v">' + (r < 0.001 ? '&lt;0.1%' : (r * 100).toFixed(1) + '%') + '</div></div>'; }
  function paint() {
    var x = chips[+a.value], y = chips[+b.value], labels = [], seen = {};
    [x, y].forEach(function (c) { c.budget.forEach(function (p) { if (!seen[p[0]]) { seen[p[0]] = 1; labels.push(p[0]); } }); });
    function get(c, l) { for (var i = 0; i < c.budget.length; i++) if (c.budget[i][0] === l) return c.budget[i][1]; return null; }
    var html = '<div class="cmpsum"><div class="k">Compute runs</div><div>' + x.runs.toLocaleString('en-US') + '</div><div>' + y.runs.toLocaleString('en-US') + '</div>' +
      '<div class="k">Silent faults found</div><div>' + x.faults + '</div><div>' + y.faults + '</div>' +
      '<div class="k">Planted faults caught</div><div>' + x.caught + ' of ' + x.injected + '</div><div>' + y.caught + ' of ' + y.injected + '</div></div>' +
      '<div class="cmphead"><div>Error budget used</div><div><a href="../chips/' + x.slug + '.html">' + esc(x.name) + '</a></div><div><a href="../chips/' + y.slug + '.html">' + esc(y.name) + '</a></div></div>';
    labels.forEach(function (l) { html += '<div class="cmprow"><div class="lab">' + esc(l) + '</div>' + cell(get(x, l)) + cell(get(y, l)) + '</div>'; });
    out.innerHTML = html.replace(/\.\.\/chips\//g, 'chips/');
  }
  a.addEventListener('change', paint); b.addEventListener('change', paint); paint();
})();
(function () {
  var f = document.getElementById('rowfilter'); if (!f) return;
  f.addEventListener('input', function () {
    var q = f.value.trim().toLowerCase();
    [].forEach.call(document.querySelectorAll('#results tbody tr'), function (tr) { tr.style.display = !q || tr.textContent.toLowerCase().indexOf(q) >= 0 ? '' : 'none'; });
  });
})();
