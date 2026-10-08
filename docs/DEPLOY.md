# chipintegrity.org, deployment notes

This folder is a complete static site. No build step, no server code, no database.

Files
- index.html      the whole site, one page, self-contained (fonts load from Google Fonts)
- robots.txt      allows all crawlers, points to the sitemap
- sitemap.xml
- llms.txt        summary for LLM crawlers
- CNAME           used only if hosted on GitHub Pages; harmless elsewhere

Hosting
- Any static host works: Nginx, Caddy, GitHub Pages, Netlify, Cloudflare Pages, S3 + CloudFront.
- Serve index.html at https://chipintegrity.org/ with HTTPS. Redirect www to the apex.
- Content-Type text/html for index.html, text/plain for llms.txt and robots.txt, application/xml for sitemap.xml.

DNS
- A records on @ to the host's IPs (for GitHub Pages: 185.199.108.153, 185.199.109.153, 185.199.110.153, 185.199.111.153)
- CNAME www to the host (for GitHub Pages: tech4biz-yasha.github.io)

Updating
- The page is generated from the result files in the repo by build_site.py (python build_site.py from the repo root, then copy docs/ to the host).
- Source: https://github.com/tech4biz-yasha/ai-chip-integrity
