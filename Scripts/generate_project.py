#!/usr/bin/env python3
"""Generate the checked-in Xcode project without third-party tooling."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
objects = {}
def ident(key): return hashlib.sha1(key.encode()).hexdigest()[:24].upper()
def add(key, value):
    objects[ident(key)] = value
    return ident(key)
def arr(values): return '(' + ', '.join(values) + (',' if values else '') + ')'
def q(value): return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'
def settings(values): return '{ ' + ' '.join(f'{k} = {v};' for k,v in values.items()) + ' }'
def config_list(key, common, debug=None, release=None):
    configs=[]
    for name, extra in [('Debug', debug or {}), ('Release', release or {})]:
        configs.append(add(f'{key}-{name}', '{ isa = XCBuildConfiguration; buildSettings = ' + settings(common | extra) + f'; name = {name}; }}'))
    return add(key+'-config-list', '{ isa = XCConfigurationList; buildConfigurations = '+arr(configs)+'; defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; }')

def file_ref(path, kind):
    return add('file-'+path, '{ isa = PBXFileReference; lastKnownFileType = '+kind+'; path = '+q(path)+'; sourceTree = SOURCE_ROOT; }')
def build_file(key, reference, extra=''):
    return add('build-'+key, '{ isa = PBXBuildFile; fileRef = '+reference+'; '+extra+' }')
def phase(key, kind, files, extra=''):
    return add(key, '{ isa = '+kind+'; buildActionMask = 2147483647; files = '+arr(files)+'; runOnlyForDeploymentPostprocessing = 0; '+extra+' }')

package = add('mlx-package', '{ isa = XCRemoteSwiftPackageReference; repositoryURL = "https://github.com/ml-explore/mlx-swift"; requirement = { kind = exactVersion; version = 0.30.6; }; }')
products={}
for target,kind,ext in [('MLXAstra','wrapper.application','app'),('MLXAstraCore','wrapper.framework','framework'),('MLXAstraTests','wrapper.cfbundle','xctest')]:
    products[target] = add('product-'+target,'{ isa = PBXFileReference; explicitFileType = '+kind+'; includeInIndex = 0; path = '+target+'.'+ext+'; sourceTree = BUILT_PRODUCTS_DIR; }')
source_paths={
    'MLXAstra': [('Sources/MLXAstra/'+f,'sourcecode.metal' if f.endswith('.metal') else 'sourcecode.swift') for f in ['MLXAstraApp.swift','SimulationModel.swift','AstraView.swift','TurbulenceCanvas.swift','Turbulence.metal','AppDiagnostics.swift']],
    'MLXAstraCore': [('Sources/MLXAstraCore/'+f,'sourcecode.swift') for f in ['SimulationTypes.swift','MLXTurbulenceSolver.swift']],
    'MLXAstraTests': [('Tests/MLXAstraTests/'+f,'sourcecode.swift') for f in ['MLXTurbulenceSolverTests.swift']],
}
asset_ref=file_ref('Resources/Assets.xcassets','folder.assetcatalog')
info_ref=file_ref('Resources/Info.plist','text.plist.xml')
groups=[]
for target,paths in source_paths.items():
    refs=[file_ref(path, kind) for path,kind in paths]
    groups.append(add('group-'+target,'{ isa = PBXGroup; children = '+arr(refs)+'; name = '+target+'; sourceTree = "<group>"; }'))
    mlx = add('mlx-product-'+target, '{ isa = XCSwiftPackageProductDependency; package = '+package+'; productName = MLX; }')
    mlx_build = add('mlx-build-'+target, '{ isa = PBXBuildFile; productRef = '+mlx+'; }')
    phases=[phase(target+'-sources','PBXSourcesBuildPhase',[build_file(target+'-'+p,r) for (p,k),r in zip(paths,refs)])]
    frameworks=[mlx_build]
    dependencies=[]
    if target != 'MLXAstraCore':
        frameworks.append(build_file(target+'-core-link',products['MLXAstraCore']))
        proxy=add(target+'-core-proxy','{ isa = PBXContainerItemProxy; containerPortal = '+ident('project')+'; proxyType = 1; remoteGlobalIDString = '+ident('target-MLXAstraCore')+'; remoteInfo = MLXAstraCore; }')
        dependencies.append(add(target+'-core-dep','{ isa = PBXTargetDependency; target = '+ident('target-MLXAstraCore')+'; targetProxy = '+proxy+'; }'))
    phases.append(phase(target+'-frameworks','PBXFrameworksBuildPhase',frameworks))
    resources=[]
    if target == 'MLXAstra':
        resources=[build_file('assets',asset_ref)]
        phases.append(phase(target+'-embed','PBXCopyFilesBuildPhase',[build_file('core-embed',products['MLXAstraCore'],'settings = { ATTRIBUTES = (CodeSignOnCopy, RemoveHeadersOnCopy,); };')],'dstPath = ""; dstSubfolderSpec = 10; name = "Embed Frameworks";'))
    phases.append(phase(target+'-resources','PBXResourcesBuildPhase',resources))
    common={'PRODUCT_NAME':'"$(TARGET_NAME)"','PRODUCT_BUNDLE_IDENTIFIER':q('com.mannetroll.'+target),'SWIFT_VERSION':'5.0','CODE_SIGN_STYLE':'Automatic','CODE_SIGN_IDENTITY':'"-"','MACOSX_DEPLOYMENT_TARGET':'14.0','SUPPORTED_PLATFORMS':'macosx','SDKROOT':'macosx','ARCHS':'arm64','ENABLE_USER_SCRIPT_SANDBOXING':'YES','SWIFT_STRICT_CONCURRENCY':'minimal','CURRENT_PROJECT_VERSION':'1','MARKETING_VERSION':'1.0','LD_RUNPATH_SEARCH_PATHS':'"$(inherited) @executable_path/../Frameworks @loader_path/../Frameworks"'}
    if target == 'MLXAstra':
        common.update({'INFOPLIST_FILE':'Resources/Info.plist','GENERATE_INFOPLIST_FILE':'NO','ASSETCATALOG_COMPILER_APPICON_NAME':'AppIcon','ENABLE_APP_SANDBOX':'NO','COMBINE_HIDPI_IMAGES':'YES'})
        ptype='com.apple.product-type.application'
    elif target == 'MLXAstraCore':
        common.update({'GENERATE_INFOPLIST_FILE':'YES','DEFINES_MODULE':'YES','SKIP_INSTALL':'YES','DYLIB_INSTALL_NAME_BASE':'"@rpath"','INSTALL_PATH':'"$(LOCAL_LIBRARY_DIR)/Frameworks"'})
        ptype='com.apple.product-type.framework'
    else:
        common.update({'GENERATE_INFOPLIST_FILE':'YES','SKIP_INSTALL':'YES','LD_RUNPATH_SEARCH_PATHS':'"$(inherited) @executable_path/../Frameworks @loader_path/../Frameworks $(BUILT_PRODUCTS_DIR)"'})
        ptype='com.apple.product-type.bundle.unit-test'
    configs=config_list(target,common,{'SWIFT_OPTIMIZATION_LEVEL':'"-Onone"','ENABLE_TESTABILITY':'YES'},{'SWIFT_OPTIMIZATION_LEVEL':'"-O"','SWIFT_COMPILATION_MODE':'wholemodule','ENABLE_TESTABILITY':'YES'})
    add('target-'+target,'{ isa = PBXNativeTarget; buildConfigurationList = '+configs+'; buildPhases = '+arr(phases)+'; buildRules = (); dependencies = '+arr(dependencies)+'; name = '+target+'; packageProductDependencies = '+arr([mlx])+'; productName = '+target+'; productReference = '+products[target]+'; productType = '+q(ptype)+'; }')

groups.append(add('resources-group','{ isa = PBXGroup; children = '+arr([asset_ref,info_ref])+'; name = Resources; sourceTree = "<group>"; }'))
prod_group=add('products-group','{ isa = PBXGroup; children = '+arr(list(products.values()))+'; name = Products; sourceTree = "<group>"; }')
groups.append(prod_group)
groups.extend([file_ref('APP.md','net.daringfireball.markdown'),file_ref('README.md','net.daringfireball.markdown')])
main_group=add('main-group','{ isa = PBXGroup; children = '+arr(groups)+'; sourceTree = "<group>"; }')
project_configs=config_list('project',{'CLANG_ENABLE_MODULES':'YES','CLANG_ENABLE_OBJC_ARC':'YES','CLANG_CXX_LANGUAGE_STANDARD':'"gnu++17"','GCC_C_LANGUAGE_STANDARD':'gnu17','MACOSX_DEPLOYMENT_TARGET':'14.0','SDKROOT':'macosx','MTL_FAST_MATH':'YES'},{'GCC_PREPROCESSOR_DEFINITIONS':'("DEBUG=1", "$(inherited)",)','DEBUG_INFORMATION_FORMAT':'dwarf','MTL_ENABLE_DEBUG_INFO':'INCLUDE_SOURCE'},{'DEBUG_INFORMATION_FORMAT':'"dwarf-with-dsym"','MTL_ENABLE_DEBUG_INFO':'NO'})
add('project','{ isa = PBXProject; attributes = { BuildIndependentTargetsInParallel = YES; LastUpgradeCheck = 2630; }; buildConfigurationList = '+project_configs+'; compatibilityVersion = "Xcode 14.0"; developmentRegion = en; hasScannedForEncodings = 0; knownRegions = (en, Base,); mainGroup = '+main_group+'; packageReferences = '+arr([package])+'; productRefGroup = '+prod_group+'; projectDirPath = ""; projectRoot = ""; targets = '+arr([ident('target-'+t) for t in products])+'; }')
project_dir=ROOT/'MLXAstra.xcodeproj'
project_dir.mkdir(exist_ok=True)
(project_dir/'project.pbxproj').write_text('// !$*UTF8*$!\n{\n archiveVersion = 1;\n classes = {};\n objectVersion = 56;\n objects = {\n'+'\n'.join(f'  {i} = {v};' for i,v in objects.items())+'\n };\n rootObject = '+ident('project')+';\n}\n')
app_id=ident('target-MLXAstra'); test_id=ident('target-MLXAstraTests')
ref=lambda i,n,p: f'<BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{i}" BuildableName="{p}" BlueprintName="{n}" ReferencedContainer="container:MLXAstra.xcodeproj"/>'
scheme=f'''<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion="2630" version="1.3">
 <BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES"><BuildActionEntries>
  <BuildActionEntry buildForTesting="YES" buildForRunning="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES">{ref(app_id,'MLXAstra','MLXAstra.app')}</BuildActionEntry>
 </BuildActionEntries></BuildAction>
 <TestAction buildConfiguration="Release" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" shouldUseLaunchSchemeArgsEnv="YES"><Testables><TestableReference skipped="NO">{ref(test_id,'MLXAstraTests','MLXAstraTests.xctest')}</TestableReference></Testables></TestAction>
 <LaunchAction buildConfiguration="Release" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" launchStyle="0" useCustomWorkingDirectory="NO" ignoresPersistentStateOnLaunch="NO" debugDocumentVersioning="YES" debugServiceExtension="internal" allowLocationSimulation="YES"><BuildableProductRunnable runnableDebuggingMode="0">{ref(app_id,'MLXAstra','MLXAstra.app')}</BuildableProductRunnable></LaunchAction>
 <ProfileAction buildConfiguration="Release" shouldUseLaunchSchemeArgsEnv="YES" savedToolIdentifier="" useCustomWorkingDirectory="NO" debugDocumentVersioning="YES"><BuildableProductRunnable runnableDebuggingMode="0">{ref(app_id,'MLXAstra','MLXAstra.app')}</BuildableProductRunnable></ProfileAction>
 <AnalyzeAction buildConfiguration="Debug"/>
 <ArchiveAction buildConfiguration="Release" revealArchiveInOrganizer="YES"/>
</Scheme>
'''
scheme_dir=project_dir/'xcshareddata/xcschemes'
scheme_dir.mkdir(parents=True,exist_ok=True)
(scheme_dir/'MLXAstra.xcscheme').write_text(scheme)
print('Generated MLXAstra.xcodeproj')
