#!/usr/bin/env lua
--[[
This requires the restserver-xavante rock to run.
run server (this file)
th tools/rest_translation_server.lua -model ../Recipes/baseline-1M-enfr/exp/model-baseline-1M-enfr_epoch13_3.44.t7 -gpuid 1
query the server:
curl -v -H "Content-Type: application/json" -X POST -d '[{ "src" : "international migration" }]' http://127.0.0.1:7784/translator/translate
]]

require('onmt.init')

local restserver = require("restserver")

local cmd = onmt.utils.ExtendedCmdLine.new('rest_targoman_server.lua')

local options = {
    {'-port', '8000', [[Port to run the server on.]]},
}

cmd:setCmdLineOptions(options, 'Server')

onmt.translate.Translator.declareOpts(cmd)

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

cmd:text("")
cmd:text("Other options")
cmd:text("")

cmd:option('-batchsize', 1000, [[Size of each parallel batch - you should not change except if low memory.]])

local opt = cmd:parse(arg)
local translator

local function tokenize(lines)
    local batch = {}

    for i = 1, #lines do
        local tokens = {}
        for word in lines[i].src:gmatch'([^%s]+)' do
            table.insert(tokens, word)
        end
        table.insert(batch, translator:buildInput(tokens))
    end

    return batch
end

local function buildResultObject(batch, rawResults)

    local results = {
        base = {},
        phrases = {},
        alignments = {}
    }

    for i = 1, #batch do
        table.insert(results.base, {
            translator:buildOutput(rawResults[i].preds[1]),
            translator:buildOutput(batch[i])
        })

        local _, wordmapping = torch.max(
            torch.cat(
                rawResults[i].preds[1].attention,
                2
            ),
            1
        )
        wordmapping = torch.totable(wordmapping:storage())
        local words = onmt.utils.Features.annotate(rawResults[i].preds[1].words, rawResults[i].preds[1].features)

        local phrases = {}
        local alignments = {}

        local j = 0
        for index in pairs(wordmapping) do
            table.insert(phrases, {
                words[j],
                j
            })
            table.insert(alignments, {
                batch[i].words[wordmapping[j]],
                j,
                {
                    { words[j], true}
                }
            })
            j = j + 1
        end
        
        table.insert(results.phrases, phrases)
        table.insert(results.alignments, alignments)
    end

    return results
end

local function translateMessage(translator, lines)

    local batch

    -- We need to tokenize the input line before translation
    _G.logger:debug("Start Tokenization")
    batch = tokenize(lines)

    -- Translate
    _G.logger:debug("Start Translation")
    local results = translator:translate(batch)
    _G.logger:debug("End Translation")

    -- Return the nbest translations for each in the batch.
    return buildResultObject(batch, results)
end

local function init_server(port, translator)
    local server = restserver:new():port(port)

    server:add_resource("translator", {
        {
        method = "POST",
        path = "/translate",
        consumes = "application/json",
        produces = "application/json",
        handler = function(req)
            _G.logger:debug("receiving request")
            local translate = translateMessage(translator, req)
            _G.logger:debug("sending response")
            return restserver.response():status(200):entity(translate)
        end,
        }
    })
    return server
end

local function main()
    -- load logger
    _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
    onmt.utils.Cuda.init(opt)

    -- disable profiling
    _G.profiler = onmt.utils.Profiler.new(false)

    _G.logger:info("Loading model")
    translator = onmt.translate.Translator.new(opt)
    _G.logger:info("Launch server")
    local server = init_server(opt.port, translator)
    -- This loads the restserver.xavante plugin
    server:enable("restserver.xavante"):start()
end

main()
