// main.bicep — Azure ML Workspace Infrastructure
// AI-300 Exam Domain 1: MLOps Infrastructure

@description('Location for all resources')
param location string = resourceGroup().location

@description('Project prefix for naming')
param projectName string = 'mlops-project'

@description('Environment tag')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: '${replace(projectName, '-', '')}${environment}st'
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: '${projectName}-${environment}-kv'
  location: location
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${projectName}-${environment}-ai'
  location: location
  kind: 'web'
  properties: { Application_Type: 'web' }
}

resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-01-01-preview' = {
  name: '${projectName}-${environment}-ws'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    storageAccount: storageAccount.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    publicNetworkAccess: 'Disabled'
  }
  tags: { environment: environment, project: projectName }
}

resource computeCluster 'Microsoft.MachineLearningServices/workspaces/computes@2024-01-01-preview' = {
  parent: mlWorkspace
  name: 'training-cluster'
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'Standard_DS3_v2'
      scaleSettings: { minNodeCount: 0, maxNodeCount: 4, nodeIdleTimeBeforeScaleDown: 'PT120S' }
    }
  }
}

output workspaceName string = mlWorkspace.name
output workspaceId string = mlWorkspace.id
output resourceGroupName string = resourceGroup().name
output computeClusterName string = computeCluster.name