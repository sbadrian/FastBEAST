
using BEAST

function hassemble(
    operator::BEAST.AbstractOperator,
    test_functions,
    trial_functions;
    compressor=:aca,
    tol=1e-4,
    treeoptions=BoxTreeOptions(nmin=100),
    maxrank=200,
    threading=:single,
    quadstrat=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
    verbose=false,
    isblockstructured=false,
    svdrecompress=false
)

    @views blkasm = BEAST.blockassembler(operator, test_functions, trial_functions)
    
    @views function assembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end

    @views farblkasm = BEAST.blockassembler(
        operator,
        test_functions,
        trial_functions,
        quadstrat=quadstrat
    )
    
    @views function farassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        farblkasm(tdata,sdata,store)
    end


    test_tree = create_tree(test_functions.pos, treeoptions)
    trial_tree = create_tree(trial_functions.pos, treeoptions)

    @time hmat = HMatrix(
        assembler,
        test_tree,
        trial_tree,
        Int64,
        scalartype(operator),
        compressor=compressor,
        tol=tol,
        maxrank=maxrank,
        threading=threading,
        farmatrixassembler=farassembler,
        verbose=verbose,
        isblockstructured=isblockstructured,
        svdrecompress=svdrecompress
    )
    return hmat
end


# The following to function ensure that no dynamic dispatching is
# performed since we know already that all triangles are well-separate

# Copied from BEAST/examples/quadstrat.jl
function BEAST.quaddata(op, tref, bref,
    tels, bels, qs::BEAST.DoubleNumQStrat)

    qs = BEAST.DoubleNumWiltonSauterQStrat(qs.outer_rule, qs.inner_rule, 1, 1, 1, 1, 1, 1)
    BEAST.quaddata(op, tref, bref, tels, bels, qs)
end

# Copied from BEAST/examples/quadstrat.jl
function BEAST.quadrule(op, tref, bref,
    i ,τ, j, σ, qd, qs::BEAST.DoubleNumQStrat)

    return BEAST.DoubleQuadRule(
        qd.tpoints[1,i],
        qd.bpoints[1,j])
end