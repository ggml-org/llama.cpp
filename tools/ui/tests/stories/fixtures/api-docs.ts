// API Documentation
export const API_DOCS_MD = String.raw`
# REST API Documentation

## 🔐 Authentication

All API requests require authentication using **Bearer tokens**. Include your API key in the Authorization header:

${'```'}http
GET /api/v1/users
Host: api.example.com
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
${'```'}

## 📍 Endpoints

### Users API

#### **GET** /api/v1/users

Retrieve a paginated list of users.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number |
| limit | integer | 20 | Items per page |
| sort | string | "created_at" | Sort field |
| order | string | "desc" | Sort order |

**Response:** 200 OK

${'```'}json
{
  "data": [
    {
      "id": "usr_1234567890",
      "email": "user@example.com",
      "name": "John Doe",
      "role": "admin",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 156,
    "pages": 8
  }
}
${'```'}

#### **POST** /api/v1/users

Create a new user account.

**Request Body:**

${'```'}json
{
  "email": "newuser@example.com",
  "password": "SecurePassword123!",
  "name": "Jane Smith",
  "role": "user"
}
${'```'}

**Response:** 201 Created

${'```'}json
{
  "id": "usr_9876543210",
  "email": "newuser@example.com",
  "name": "Jane Smith",
  "role": "user",
  "created_at": "2024-01-21T09:15:00Z"
}
${'```'}

### Error Responses

The API returns errors in a consistent format:

${'```'}json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {
        "field": "email",
        "message": "Email format is invalid"
      }
    ]
  }
}
${'```'}

### Rate Limiting

| Tier | Requests/Hour | Burst |
|------|--------------|-------|
| **Free** | 1,000 | 100 |
| **Pro** | 10,000 | 500 |
| **Enterprise** | Unlimited | - |

**Headers:**
- X-RateLimit-Limit
- X-RateLimit-Remaining  
- X-RateLimit-Reset

### Webhooks

Configure webhooks to receive real-time events:

${'```'}javascript
// Webhook payload
{
  "event": "user.created",
  "timestamp": "2024-01-21T09:15:00Z",
  "data": {
    "id": "usr_9876543210",
    "email": "newuser@example.com"
  },
  "signature": "sha256=abcd1234..."
}
${'```'}

### SDK Examples

**JavaScript/TypeScript:**

${'```'}typescript
import { ApiClient } from '@example/api-sdk';

const client = new ApiClient({
  apiKey: process.env.API_KEY
});

const users = await client.users.list({
  page: 1,
  limit: 20
});
${'```'}

**Python:**

${'```'}python
from example_api import Client

client = Client(api_key=os.environ['API_KEY'])
users = client.users.list(page=1, limit=20)
${'```'}

---

📚 [Full API Reference](https://api.example.com/docs) | 💬 [Support](https://support.example.com)
`;
