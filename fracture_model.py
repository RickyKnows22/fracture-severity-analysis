# # import torch
# # import torch.nn as nn
# # import timm

# # class FracturePatternModel(nn.Module):
# #     def __init__(self, pretrained=True):
# #         super(FracturePatternModel, self).__init__()
        
# #         # Use EfficientNetV2 as backbone with focus on feature extraction
# #         self.backbone = timm.create_model(
# #             'tf_efficientnetv2_m',
# #             pretrained=pretrained,
# #             in_chans=1,  # Grayscale input
# #             num_classes=0,
# #             global_pool=''
# #         )
        
# #         feat_dim = self.backbone.num_features
        
# #         # Multi-head classifier for different aspects of fracture classification
# #         self.shared_features = nn.Sequential(
# #             nn.AdaptiveAvgPool2d((1, 1)),
# #             nn.Flatten(),
# #             nn.Linear(feat_dim, 1024),
# #             nn.LayerNorm(1024),
# #             nn.GELU(),
# #             nn.Dropout(0.3)
# #         )
        
# #         # Pattern classification head (Simple, Wedge, Comminuted)
# #         self.pattern_classifier = nn.Sequential(
# #             nn.Linear(1024, 512),
# #             nn.LayerNorm(512),
# #             nn.GELU(),
# #             nn.Dropout(0.2),
# #             nn.Linear(512, 3)  # 3 pattern classes
# #         )
        
# #         # Joint involvement binary classifier
# #         self.joint_classifier = nn.Sequential(
# #             nn.Linear(1024, 256),
# #             nn.LayerNorm(256),
# #             nn.GELU(),
# #             nn.Dropout(0.2),
# #             nn.Linear(256, 1)  # Binary classification
# #         )
        
# #         # Displacement assessment head
# #         self.displacement_classifier = nn.Sequential(
# #             nn.Linear(1024, 256),
# #             nn.LayerNorm(256),
# #             nn.GELU(),
# #             nn.Dropout(0.2),
# #             nn.Linear(256, 3)  # Minimal, Moderate, Severe displacement
# #         )
        
# #         self._initialize_weights()
    
# #     def _initialize_weights(self):
# #         for m in self.modules():
# #             if isinstance(m, nn.Linear):
# #                 nn.init.kaiming_normal_(m.weight)
# #                 if m.bias is not None:
# #                     nn.init.constant_(m.bias, 0)
    
# #     def forward(self, x):
# #         features = self.backbone(x)
# #         shared = self.shared_features(features)
        
# #         pattern = self.pattern_classifier(shared)
# #         joint = self.joint_classifier(shared)  # Remove sigmoid, it's included in BCEWithLogitsLoss
# #         displacement = self.displacement_classifier(shared)
        
# #         return {
# #             'pattern': pattern,
# #             'joint_involvement': joint,
# #             'displacement': displacement
# #         }

# # class FractureLoss(nn.Module):
# #     def __init__(self, pattern_weight=1.0, joint_weight=0.5, displacement_weight=0.5):
# #         super().__init__()
# #         self.pattern_weight = pattern_weight
# #         self.joint_weight = joint_weight
# #         self.displacement_weight = displacement_weight
        
# #         # Calculate class weights for pattern classification
# #         pattern_class_weights = torch.tensor([
# #             1000.0,  # Simple - highest weight due to few samples
# #             2000.0,  # Wedge - even higher weight due to very few samples
# #             1.0      # Comminuted - lowest weight as it's the majority
# #         ])
        
# #         self.pattern_criterion = nn.CrossEntropyLoss(weight=pattern_class_weights)
# #         self.joint_criterion = nn.BCEWithLogitsLoss()
# #         self.displacement_criterion = nn.CrossEntropyLoss()
    
# #     def to(self, device):
# #         super().to(device)
# #         self.pattern_criterion.weight = self.pattern_criterion.weight.to(device)
# #         return self
    
# #     def forward(self, predictions, targets):
# #         # Ensure all inputs are the correct shape and type
# #         pattern_pred = predictions['pattern']
# #         joint_pred = predictions['joint_involvement']
# #         displacement_pred = predictions['displacement']
        
# #         pattern_target = targets['pattern']
# #         joint_target = targets['joint_involvement'].float()
# #         displacement_target = targets['displacement']
        
# #         # Calculate individual losses
# #         pattern_loss = self.pattern_criterion(pattern_pred, pattern_target)
# #         joint_loss = self.joint_criterion(joint_pred, joint_target)
# #         displacement_loss = self.displacement_criterion(displacement_pred, displacement_target)
        
# #         # Weighted sum
# #         total_loss = (
# #             self.pattern_weight * pattern_loss +
# #             self.joint_weight * joint_loss +
# #             self.displacement_weight * displacement_loss
# #         )
        
# #         # Return total loss and individual components
# #         loss_dict = {
# #             'total': total_loss.item(),
# #             'pattern': pattern_loss.item(),
# #             'joint': joint_loss.item(),
# #             'displacement': displacement_loss.item()
# #         }
        
# #         return total_loss, loss_dict


# import torch
# import torch.nn as nn
# import timm

# class FracturePatternModel(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
        
#         # Use EfficientNetV2 backbone with high-resolution features
#         self.backbone = timm.create_model(
#             'tf_efficientnetv2_m',
#             pretrained=pretrained,
#             in_chans=1,
#             features_only=True,  # Get intermediate feature maps
#             out_indices=(2, 3, 4)  # Multiple scales for better pattern detection
#         )
        
#         # Feature Pyramid Network for multi-scale analysis
#         self.fpn = FeaturePyramidNetwork(
#             in_channels_list=self.backbone.feature_info.channels()[-3:],
#             out_channels=256
#         )
        
#         # Pattern analysis module
#         self.pattern_module = nn.Sequential(
#             nn.AdaptiveAvgPool2d((7, 7)),
#             nn.Flatten(),
#             nn.Linear(256 * 7 * 7, 1024),
#             nn.LayerNorm(1024),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.LayerNorm(512),
#             nn.GELU(),
#             nn.Dropout(0.2)
#         )
        
#         # Classification heads
#         self.pattern_head = nn.Linear(512, 3)  # Simple, Wedge, Comminuted
#         self.joint_head = nn.Linear(512, 1)    # Joint involvement
#         self.displacement_head = nn.Linear(512, 3)  # Displacement severity
        
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=0.01)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
    
#     def forward(self, x):
#         # Extract multi-scale features
#         features = self.backbone(x)
        
#         # FPN for feature refinement
#         fpn_features = self.fpn(features)
        
#         # Pattern analysis
#         pattern_features = self.pattern_module(fpn_features['0'])
        
#         # Generate predictions
#         pattern_out = self.pattern_head(pattern_features)
#         joint_out = self.joint_head(pattern_features)
#         displacement_out = self.displacement_head(pattern_features)
        
#         return {
#             'pattern': pattern_out,
#             'joint_involvement': joint_out,
#             'displacement': displacement_out
#         }

# class FeaturePyramidNetwork(nn.Module):
#     def __init__(self, in_channels_list, out_channels):
#         super().__init__()
        
#         self.inner_blocks = nn.ModuleList()
#         self.layer_blocks = nn.ModuleList()
        
#         for in_channels in in_channels_list:
#             inner_block = nn.Conv2d(in_channels, out_channels, 1)
#             layer_block = nn.Sequential(
#                 nn.Conv2d(out_channels, out_channels, 3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True)
#             )
#             self.inner_blocks.append(inner_block)
#             self.layer_blocks.append(layer_block)
    
#     def forward(self, features):
#         last_inner = self.inner_blocks[-1](features[-1])
#         results = []
#         results.append(self.layer_blocks[-1](last_inner))
        
#         for idx in range(len(features) - 2, -1, -1):
#             inner_lateral = self.inner_blocks[idx](features[idx])
#             feat_shape = inner_lateral.shape[-2:]
#             inner_top_down = nn.functional.interpolate(
#                 last_inner, size=feat_shape, mode="nearest"
#             )
#             last_inner = inner_lateral + inner_top_down
#             results.insert(0, self.layer_blocks[idx](last_inner))
        
#         return {str(idx): feature for idx, feature in enumerate(results)}

# class FractureLoss(nn.Module):
#     def __init__(self, pattern_weight=1.0, joint_weight=0.5, displacement_weight=0.5):
#         super().__init__()
#         self.pattern_weight = pattern_weight
#         self.joint_weight = joint_weight
#         self.displacement_weight = displacement_weight
        
#         # Calculate class weights for pattern classification
#         pattern_class_weights = torch.tensor([
#             1.0,    # Simple
#             2.0,    # Wedge - higher weight as it's less common
#             1.5     # Comminuted
#         ])
        
#         self.pattern_criterion = nn.CrossEntropyLoss(weight=pattern_class_weights)
#         self.joint_criterion = nn.BCEWithLogitsLoss()
#         self.displacement_criterion = nn.CrossEntropyLoss()
    
#     def to(self, device):
#         super().to(device)
#         self.pattern_criterion.weight = self.pattern_criterion.weight.to(device)
#         return self
    
#     def forward(self, predictions, targets):
#         # Handle dictionary outputs from model
#         if isinstance(predictions, dict):
#             pattern_pred = predictions['pattern']
#             joint_pred = predictions['joint_involvement']
#             displacement_pred = predictions['displacement']
#         else:
#             raise TypeError("Expected model outputs to be a dictionary")

#         # Extract targets
#         pattern_target = targets['pattern']
#         joint_target = targets['joint_involvement'].float()
#         displacement_target = targets['displacement']

#         # Calculate individual losses
#         pattern_loss = self.pattern_criterion(pattern_pred, pattern_target)
#         joint_loss = self.joint_criterion(joint_pred, joint_target)
#         displacement_loss = self.displacement_criterion(displacement_pred, displacement_target)
        
#         # Calculate weighted sum
#         total_loss = (
#             self.pattern_weight * pattern_loss +
#             self.joint_weight * joint_loss +
#             self.displacement_weight * displacement_loss
#         )
        
#         return total_loss, {
#             'pattern_loss': pattern_loss.item(),
#             'joint_loss': joint_loss.item(),
#             'displacement_loss': displacement_loss.item()
#         }


# import torch
# import torch.nn as nn
# import timm

# class FracturePatternModel(nn.Module):
#     def __init__(self, pretrained=True):
#         super(FracturePatternModel, self).__init__()
        
#         # Use EfficientNetV2 as backbone with focus on feature extraction
#         self.backbone = timm.create_model(
#             'tf_efficientnetv2_m',
#             pretrained=pretrained,
#             in_chans=1,  # Grayscale input
#             num_classes=0,
#             global_pool=''
#         )
        
#         feat_dim = self.backbone.num_features
        
#         # Multi-head classifier for different aspects of fracture classification
#         self.shared_features = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(feat_dim, 1024),
#             nn.LayerNorm(1024),
#             nn.GELU(),
#             nn.Dropout(0.3)
#         )
        
#         # Pattern classification head (Simple, Wedge, Comminuted)
#         self.pattern_classifier = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.LayerNorm(512),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 3)  # 3 pattern classes
#         )
        
#         # Joint involvement binary classifier
#         self.joint_classifier = nn.Sequential(
#             nn.Linear(1024, 256),
#             nn.LayerNorm(256),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 1)  # Binary classification
#         )
        
#         # Displacement assessment head
#         self.displacement_classifier = nn.Sequential(
#             nn.Linear(1024, 256),
#             nn.LayerNorm(256),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 3)  # Minimal, Moderate, Severe displacement
#         )
        
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
    
#     def forward(self, x):
#         features = self.backbone(x)
#         shared = self.shared_features(features)
        
#         pattern = self.pattern_classifier(shared)
#         joint = self.joint_classifier(shared)  # Remove sigmoid, it's included in BCEWithLogitsLoss
#         displacement = self.displacement_classifier(shared)
        
#         return {
#             'pattern': pattern,
#             'joint_involvement': joint,
#             'displacement': displacement
#         }

# class FractureLoss(nn.Module):
#     def __init__(self, pattern_weight=1.0, joint_weight=0.5, displacement_weight=0.5):
#         super().__init__()
#         self.pattern_weight = pattern_weight
#         self.joint_weight = joint_weight
#         self.displacement_weight = displacement_weight
        
#         # Calculate class weights for pattern classification
#         pattern_class_weights = torch.tensor([
#             1000.0,  # Simple - highest weight due to few samples
#             2000.0,  # Wedge - even higher weight due to very few samples
#             1.0      # Comminuted - lowest weight as it's the majority
#         ])
        
#         self.pattern_criterion = nn.CrossEntropyLoss(weight=pattern_class_weights)
#         self.joint_criterion = nn.BCEWithLogitsLoss()
#         self.displacement_criterion = nn.CrossEntropyLoss()
    
#     def to(self, device):
#         super().to(device)
#         self.pattern_criterion.weight = self.pattern_criterion.weight.to(device)
#         return self
    
#     def forward(self, predictions, targets):
#         # Ensure all inputs are the correct shape and type
#         pattern_pred = predictions['pattern']
#         joint_pred = predictions['joint_involvement']
#         displacement_pred = predictions['displacement']
        
#         pattern_target = targets['pattern']
#         joint_target = targets['joint_involvement'].float()
#         displacement_target = targets['displacement']
        
#         # Calculate individual losses
#         pattern_loss = self.pattern_criterion(pattern_pred, pattern_target)
#         joint_loss = self.joint_criterion(joint_pred, joint_target)
#         displacement_loss = self.displacement_criterion(displacement_pred, displacement_target)
        
#         # Weighted sum
#         total_loss = (
#             self.pattern_weight * pattern_loss +
#             self.joint_weight * joint_loss +
#             self.displacement_weight * displacement_loss
#         )
        
#         # Return total loss and individual components
#         loss_dict = {
#             'total': total_loss.item(),
#             'pattern': pattern_loss.item(),
#             'joint': joint_loss.item(),
#             'displacement': displacement_loss.item()
#         }
        
#         return total_loss, loss_dict


import torch
import torch.nn as nn
import timm

class FracturePatternModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Use EfficientNetV2 backbone with high-resolution features
        self.backbone = timm.create_model(
            'tf_efficientnetv2_m',
            pretrained=pretrained,
            in_chans=1,
            features_only=True,  # Get intermediate feature maps
            out_indices=(2, 3, 4)  # Multiple scales for better pattern detection
        )
        
        # Feature Pyramid Network for multi-scale analysis
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.backbone.feature_info.channels()[-3:],
            out_channels=256
        )
        
        # Pattern analysis module
        self.pattern_module = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Classification heads
        self.pattern_head = nn.Linear(512, 3)  # Simple, Wedge, Comminuted
        self.joint_head = nn.Linear(512, 1)    # Joint involvement
        self.displacement_head = nn.Linear(512, 3)  # Displacement severity
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)
        
        # FPN for feature refinement
        fpn_features = self.fpn(features)
        
        # Pattern analysis
        pattern_features = self.pattern_module(fpn_features['0'])
        
        # Generate predictions
        pattern_out = self.pattern_head(pattern_features)
        joint_out = self.joint_head(pattern_features)
        displacement_out = self.displacement_head(pattern_features)
        
        return {
            'pattern': pattern_out,
            'joint_involvement': joint_out,
            'displacement': displacement_out
        }

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
    
    def forward(self, features):
        last_inner = self.inner_blocks[-1](features[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))
        
        for idx in range(len(features) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](features[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = nn.functional.interpolate(
                last_inner, size=feat_shape, mode="nearest"
            )
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
        
        return {str(idx): feature for idx, feature in enumerate(results)}

class FractureLoss(nn.Module):
    def __init__(self, pattern_weight=1.0, joint_weight=0.5, class_weights=None):
        super().__init__()
        self.pattern_weight = pattern_weight
        self.joint_weight = joint_weight
        
        # Initialize the loss functions with class weights if provided
        self.pattern_criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        self.joint_criterion = nn.BCEWithLogitsLoss()
    
    def to(self, device):
        super().to(device)
        self.pattern_criterion.weight = self.pattern_criterion.weight.to(device)
        return self
    
    def forward(self, predictions, targets):
        # Ensure all inputs are the correct shape and type
        pattern_pred = predictions['pattern']
        joint_pred = predictions['joint_involvement']
        
        pattern_target = targets['pattern']
        joint_target = targets['joint_involvement'].float()
        
        # Calculate individual losses
        pattern_loss = self.pattern_criterion(pattern_pred, pattern_target)
        joint_loss = self.joint_criterion(joint_pred, joint_target)
        
        # Calculate weighted sum
        total_loss = self.pattern_weight * pattern_loss + self.joint_weight * joint_loss
        
        # Return total loss and individual components
        loss_dict = {
            'total': total_loss.item(),
            'pattern': pattern_loss.item(),
            'joint': joint_loss.item()
        }
        
        return total_loss, loss_dict