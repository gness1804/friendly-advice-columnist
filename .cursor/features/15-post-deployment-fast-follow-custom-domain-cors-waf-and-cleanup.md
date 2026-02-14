# Post Deployment Fast Follow Custom Domain Cors Waf And Cleanup

## Working directory

`~/Desktop/friendly-advice-columnist`

## Contents

## Working directory

`~/Desktop/friendly-advice-columnist`

## Contents

**Branch:** `feature/aws-deployment`
**App Runner URL:** https://qejunep2xs.us-east-2.awsapprunner.com
**Status:** App deployed and running successfully

### Remaining steps

1. **Set up custom domain** — follow instructions in `docs/DEPLOYMENT.md` (DNS CNAME records, certificate validation)
2. **Update `ALLOWED_ORIGINS`** env var in App Runner to include the custom domain
3. **Set up AWS WAF** — App Runner supports WAF for additional DDoS/bot protection. Configure at the AWS level (not in code) for rate limiting, IP filtering, and bot mitigation.
4. **Merge `feature/aws-deployment` branch to master** and create a PR
5. **Resolve CFS/GitHub sync conflict on progress/7** — run `cfs gh sync` interactively

## Acceptance criteria

- Custom domain is configured and serving the app
- `ALLOWED_ORIGINS` is updated with the custom domain
- AWS WAF is evaluated and optionally configured
- Branch is merged to master via PR

## Acceptance criteria
