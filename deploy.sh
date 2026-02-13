#!/usr/bin/env bash
#
# Deploy Friendly Advice Columnist to AWS App Runner.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker installed and running
#
# Usage:
#   ./deploy.sh                   # Full deploy (build, push, create/update service)
#   ./deploy.sh --build-only      # Build and push Docker image only
#   ./deploy.sh --dry-run         # Show what would be done without executing
#
# Required environment variables (set in .env or export before running):
#   OWNER_API_KEY_HASH    - SHA-256 hash of the owner's OpenAI API key
#
# Optional environment variables:
#   AWS_REGION            - AWS region (default: us-east-2)
#   AWS_ACCOUNT_ID        - Auto-detected if not set
#   APP_NAME              - App name (default: friendly-advice-columnist)
#   CUSTOM_DOMAIN         - Custom domain for the service (optional)

set -euo pipefail

# ---------- Configuration ----------
AWS_REGION="${AWS_REGION:-us-east-2}"
APP_NAME="${APP_NAME:-friendly-advice-columnist}"
ECR_REPO_NAME="${APP_NAME}"
DYNAMODB_TABLE="${DYNAMODB_TABLE:-friendly-advice-conversations}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DRY_RUN=false
BUILD_ONLY=false

# Parse flags
for arg in "$@"; do
    case $arg in
        --dry-run)    DRY_RUN=true ;;
        --build-only) BUILD_ONLY=true ;;
        *)            echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# Auto-detect AWS account ID
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo "=== Friendly Advice Columnist Deployment ==="
echo "Region:       ${AWS_REGION}"
echo "Account:      ${AWS_ACCOUNT_ID}"
echo "ECR Repo:     ${ECR_URI}"
echo "Image Tag:    ${IMAGE_TAG}"
echo "DynamoDB:     ${DYNAMODB_TABLE}"
echo "Dry Run:      ${DRY_RUN}"
echo ""

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $*"
    else
        echo ">>> $*"
        "$@"
    fi
}

# ---------- Step 1: Create ECR repository (if it doesn't exist) ----------
echo "--- Step 1: ECR Repository ---"
if ! aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" > /dev/null 2>&1; then
    run_cmd aws ecr create-repository \
        --repository-name "${ECR_REPO_NAME}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true
else
    echo "ECR repository already exists."
fi

# ---------- Step 2: Build Docker image ----------
echo ""
echo "--- Step 2: Build Docker Image ---"
run_cmd docker build -t "${ECR_REPO_NAME}:${IMAGE_TAG}" .

# ---------- Step 3: Push to ECR ----------
echo ""
echo "--- Step 3: Push to ECR ---"
run_cmd aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
run_cmd docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
run_cmd docker push "${ECR_URI}:${IMAGE_TAG}"

if [ "$BUILD_ONLY" = true ]; then
    echo ""
    echo "=== Build complete. Image pushed to ${ECR_URI}:${IMAGE_TAG} ==="
    exit 0
fi

# ---------- Step 4: Create DynamoDB table (if it doesn't exist) ----------
echo ""
echo "--- Step 4: DynamoDB Table ---"
if ! aws dynamodb describe-table --table-name "${DYNAMODB_TABLE}" --region "${AWS_REGION}" > /dev/null 2>&1; then
    run_cmd aws dynamodb create-table \
        --table-name "${DYNAMODB_TABLE}" \
        --attribute-definitions \
            AttributeName=user_id,AttributeType=S \
            AttributeName=session_id,AttributeType=S \
        --key-schema \
            AttributeName=user_id,KeyType=HASH \
            AttributeName=session_id,KeyType=RANGE \
        --billing-mode PAY_PER_REQUEST \
        --region "${AWS_REGION}"

    echo "Waiting for table to become active..."
    run_cmd aws dynamodb wait table-exists --table-name "${DYNAMODB_TABLE}" --region "${AWS_REGION}"

    # Enable TTL
    run_cmd aws dynamodb update-time-to-live \
        --table-name "${DYNAMODB_TABLE}" \
        --time-to-live-specification "Enabled=true, AttributeName=ttl" \
        --region "${AWS_REGION}"
else
    echo "DynamoDB table already exists."
fi

# ---------- Step 5: Create IAM role for App Runner (if it doesn't exist) ----------
echo ""
echo "--- Step 5: IAM Role ---"
ROLE_NAME="${APP_NAME}-apprunner-role"
INSTANCE_ROLE_NAME="${APP_NAME}-instance-role"

# Access role (for ECR)
if ! aws iam get-role --role-name "${ROLE_NAME}" > /dev/null 2>&1; then
    run_cmd aws iam create-role \
        --role-name "${ROLE_NAME}" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "build.apprunner.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }'
    run_cmd aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
else
    echo "Access role already exists."
fi

# Instance role (for DynamoDB access)
if ! aws iam get-role --role-name "${INSTANCE_ROLE_NAME}" > /dev/null 2>&1; then
    run_cmd aws iam create-role \
        --role-name "${INSTANCE_ROLE_NAME}" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "tasks.apprunner.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }'

    # Inline policy for DynamoDB access (least privilege)
    run_cmd aws iam put-role-policy \
        --role-name "${INSTANCE_ROLE_NAME}" \
        --policy-name "dynamodb-access" \
        --policy-document "{
            \"Version\": \"2012-10-17\",
            \"Statement\": [{
                \"Effect\": \"Allow\",
                \"Action\": [
                    \"dynamodb:PutItem\",
                    \"dynamodb:GetItem\",
                    \"dynamodb:Query\",
                    \"dynamodb:DeleteItem\",
                    \"dynamodb:BatchWriteItem\"
                ],
                \"Resource\": \"arn:aws:dynamodb:${AWS_REGION}:${AWS_ACCOUNT_ID}:table/${DYNAMODB_TABLE}\"
            }]
        }"
else
    echo "Instance role already exists."
fi

ACCESS_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"
INSTANCE_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${INSTANCE_ROLE_NAME}"

# ---------- Step 6: Create or update App Runner service ----------
echo ""
echo "--- Step 6: App Runner Service ---"

# Check required env var
if [ -z "${OWNER_API_KEY_HASH:-}" ]; then
    echo "WARNING: OWNER_API_KEY_HASH is not set. The fine-tuned model will not be used for any user."
    echo "Generate it with: python -c \"import hashlib; print(hashlib.sha256(b'your-api-key').hexdigest())\""
fi

SERVICE_EXISTS=$(aws apprunner list-services --region "${AWS_REGION}" --query "ServiceSummaryList[?ServiceName=='${APP_NAME}'].ServiceArn" --output text 2>/dev/null || echo "")

if [ -z "$SERVICE_EXISTS" ]; then
    echo "Creating new App Runner service..."
    run_cmd aws apprunner create-service \
        --service-name "${APP_NAME}" \
        --region "${AWS_REGION}" \
        --source-configuration "{
            \"AuthenticationConfiguration\": {
                \"AccessRoleArn\": \"${ACCESS_ROLE_ARN}\"
            },
            \"ImageRepository\": {
                \"ImageIdentifier\": \"${ECR_URI}:${IMAGE_TAG}\",
                \"ImageRepositoryType\": \"ECR\",
                \"ImageConfiguration\": {
                    \"Port\": \"8000\",
                    \"RuntimeEnvironmentVariables\": {
                        \"OWNER_API_KEY_HASH\": \"${OWNER_API_KEY_HASH:-}\",
                        \"DYNAMODB_TABLE\": \"${DYNAMODB_TABLE}\",
                        \"AWS_REGION\": \"${AWS_REGION}\",
                        \"ALLOWED_ORIGINS\": \"${CUSTOM_DOMAIN:-}\"
                    }
                }
            },
            \"AutoDeploymentsEnabled\": false
        }" \
        --instance-configuration "{
            \"Cpu\": \"1024\",
            \"Memory\": \"2048\",
            \"InstanceRoleArn\": \"${INSTANCE_ROLE_ARN}\"
        }" \
        --health-check-configuration "{
            \"Protocol\": \"HTTP\",
            \"Path\": \"/health\",
            \"Interval\": 10,
            \"Timeout\": 5,
            \"HealthyThreshold\": 1,
            \"UnhealthyThreshold\": 5
        }"
else
    echo "Updating existing App Runner service..."
    SERVICE_ARN="${SERVICE_EXISTS}"
    run_cmd aws apprunner update-service \
        --service-arn "${SERVICE_ARN}" \
        --region "${AWS_REGION}" \
        --source-configuration "{
            \"AuthenticationConfiguration\": {
                \"AccessRoleArn\": \"${ACCESS_ROLE_ARN}\"
            },
            \"ImageRepository\": {
                \"ImageIdentifier\": \"${ECR_URI}:${IMAGE_TAG}\",
                \"ImageRepositoryType\": \"ECR\",
                \"ImageConfiguration\": {
                    \"Port\": \"8000\",
                    \"RuntimeEnvironmentVariables\": {
                        \"OWNER_API_KEY_HASH\": \"${OWNER_API_KEY_HASH:-}\",
                        \"DYNAMODB_TABLE\": \"${DYNAMODB_TABLE}\",
                        \"AWS_REGION\": \"${AWS_REGION}\",
                        \"ALLOWED_ORIGINS\": \"${CUSTOM_DOMAIN:-}\"
                    }
                }
            },
            \"AutoDeploymentsEnabled\": false
        }"
fi

echo ""
echo "=== Deployment initiated! ==="
echo ""
echo "Check service status with:"
echo "  aws apprunner list-services --region ${AWS_REGION} --query 'ServiceSummaryList[?ServiceName==\`${APP_NAME}\`]'"
echo ""
echo "Get service URL with:"
echo "  aws apprunner describe-service --service-arn \$(aws apprunner list-services --region ${AWS_REGION} --query 'ServiceSummaryList[?ServiceName==\`${APP_NAME}\`].ServiceArn' --output text) --region ${AWS_REGION} --query 'Service.ServiceUrl' --output text"
