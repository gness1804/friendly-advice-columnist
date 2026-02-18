# Deployment Guide

Deploy the Friendly Advice Columnist to AWS using App Runner, ECR, and DynamoDB.

## Architecture

- **AWS App Runner**: Hosts the containerized FastAPI app with auto-scaling and HTTPS
- **Amazon ECR**: Docker image registry
- **Amazon DynamoDB**: Serverless conversation persistence (90-day TTL)
- **IAM Roles**: Least-privilege access for ECR pull and DynamoDB operations

## Prerequisites

- AWS CLI configured (`aws configure`) with an IAM user that has admin or appropriate permissions
- Docker installed and running
- Python 3.10+

## Quick Start

### 1. Generate your owner API key hash

This hash lets the app recognize your API key and use the fine-tuned model:

```bash
python -c "import hashlib; print(hashlib.sha256(b'sk-your-actual-key').hexdigest())"
```

### 2. Set environment variables

```bash
export OWNER_API_KEY_HASH="<hash from step 1>" && export AWS_REGION="us-east-2"
```

### 3. Deploy

```bash
# Dry run first to see what will happen
./deploy.sh --dry-run

# Full deployment
./deploy.sh
```

The script will:
1. Create an ECR repository (if needed)
2. Build and push the Docker image
3. Create the DynamoDB table with TTL enabled (if needed)
4. Create IAM roles with least-privilege policies (if needed)
5. Create or update the App Runner service

### 4. Get your service URL

```bash
aws apprunner list-services --region us-east-2 --query 'ServiceSummaryList[?ServiceName==`friendly-advice-columnist`].ServiceUrl' --output text
```

## Custom Domain

Associate a custom domain directly with App Runner:

1. Associate your domain in the App Runner console or via CLI:
   ```bash
   aws apprunner associate-custom-domain --service-arn <service-arn> --domain-name your-domain.com --region us-east-2
   ```
2. Add the provided CNAME records to your DNS provider
3. Wait for certificate validation (can take up to 48 hours)
4. Update `ALLOWED_ORIGINS` env var in the App Runner service to include your domain

## WAF (Recommended)

App Runner supports associating an AWS WAF Web ACL directly (no CloudFront required).

### 1. Create and associate a Web ACL

1. Open **App Runner** (region: `us-east-2`).
2. Select the service (default: `friendly-advice-columnist`).
3. Open the **Security** tab.
4. Under **Web application firewall (AWS WAF)**, click **Associate web ACL**.
5. Choose **Create web ACL** and fill in:
   - Name: `friendly-advice-columnist-waf`
   - Resource type: `App Runner`
   - Scope: `Regional`
   - Default action: `Allow`

### 2. Add baseline managed rules

Add these AWS Managed Rule Groups:
- `AWSManagedRulesCommonRuleSet`
- `AWSManagedRulesKnownBadInputsRuleSet`
- `AWSManagedRulesAmazonIpReputationList`
- `AWSManagedRulesAnonymousIpList`
- `AWSManagedRulesSQLiRuleSet`

### 3. Add a rate-based rule

Add a rate-based rule to slow abusive clients:
- Name: `RateLimit-IP`
- Limit: `2000` requests per `5` minutes per IP (tune later)
- Action: `Block`

### 4. Verify

1. Open **WAF & Shield** → **Web ACLs** → `friendly-advice-columnist-waf`.
2. Confirm the App Runner service is listed under **Associated AWS resources**.
3. Check **Monitoring** for request metrics.

Optional: enable WAF logging to CloudWatch Logs or S3.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OWNER_API_KEY_HASH` | Yes | - | SHA-256 hash of the owner's OpenAI API key |
| `AWS_REGION` | No | `us-east-2` | AWS region |
| `DYNAMODB_TABLE` | No | `friendly-advice-conversations` | DynamoDB table name |
| `CONVERSATION_TTL_DAYS` | No | `90` | Days before conversations auto-expire |
| `ALLOWED_ORIGINS` | No | - | Comma-separated allowed CORS origins |
| `BASE_MODEL` | No | `gpt-4.1-mini` | OpenAI base model |
| `FINE_TUNED_MODEL` | No | (project default) | OpenAI fine-tuned model ID |

## How API Key Routing Works

- Users enter their OpenAI API key in the browser (stored in localStorage, sent per-request)
- The backend hashes the provided key and compares it to `OWNER_API_KEY_HASH`
- **Owner's key**: Full two-stage pipeline (base model draft + fine-tuned model revision)
- **Other keys**: Single-stage pipeline (base model only, using the columnist system prompt)
- No API keys are stored server-side

## Security Features

- **Rate limiting**: 10 requests/minute per IP on advice endpoints
- **Security headers**: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy, Permissions-Policy
- **XSS protection**: All response content is HTML-escaped before rendering
- **CORS**: Configurable allowed origins
- **Input validation**: Max question length, question relevance screening
- **DynamoDB**: User data is keyed by hashed API key (not the key itself)

## Updating the Deployment

Build and push a new image, then update the service:

```bash
./deploy.sh
```

Or build only (without updating the service):

```bash
./deploy.sh --build-only
```

## Cost Estimate

With light usage (< 100 requests/day):
- **App Runner**: ~$5-10/month (1 vCPU, 2GB RAM, auto-pauses when idle)
- **DynamoDB**: < $1/month (on-demand pricing)
- **ECR**: < $1/month (image storage)
